#include "exateppabm/population.h"

#include <fmt/core.h>

#include <algorithm>
#include <numeric>
#include <memory>
#include <random>
#include <array>
#include <vector>

#include "exateppabm/demographics.h"
#include "exateppabm/disease.h"
#include "exateppabm/person.h"
#include "exateppabm/input.h"
#include "exateppabm/util.h"
#include "exateppabm/visualisation.h"

namespace exateppabm {
namespace population {

namespace {

// File-scoped array  containing the number of infected agents per demographic from population initialisation. This needs to be made accessible to a FLAME GPU Init func due to macro environment property limitations.

std::array<std::uint64_t, demographics::AGE_COUNT> _infectedPerDemographic = {};

}  // namespace

std::unique_ptr<flamegpu::AgentVector> generate(flamegpu::ModelDescription& model, const exateppabm::input::config params, const bool verbose) {
    fmt::print("@todo - validate params inputs when generated agents (pop size, initial infected count etc)\n");

    // @todo - assert that the requested initial population is non zero.
    auto pop = std::make_unique<flamegpu::AgentVector>(model.Agent(exateppabm::person::NAME), params.n_total);

    // seed host rng for population generation.
    // @todo - does this want to be a separate seed from the config file?
    std::mt19937_64 rng(params.rng_seed);

    // Need to initialise a fixed number of individuals as infected.
    // This not very scalable way of doing it, is to create a vector with one element per individual in the simulation, initialised to false
    // set the first n_seed_infection elements to true/1
    // Shuffle the vector,  and query at agent creation time
    // RNG sampling in-loop would be more memory efficient, but harder to guarantee that exactly enough are created. This will likely be replaced anyway, so quick and dirty is fine.
    std::vector<bool> infected_vector(params.n_total);
    std::fill(infected_vector.begin(), infected_vector.begin() + std::min(params.n_total, params.n_seed_infection), true);
    std::shuffle(infected_vector.begin(), infected_vector.end(), rng);

    // Get the number of individuals per house and their age demographics
    auto households = generateHouseholdStructures(params, rng, verbose);

    // per demo total is not an output in time series.
    // Alternately, we need to initialise the exact number of each age band, not RNG, and just scale it down accordingly. Will look at in "realistic" population generation
    std::array<std::uint64_t, demographics::AGE_COUNT> createdPerDemographic = {};
    for (const auto & household : households) {
        for (demographics::AgeUnderlyingType i = 0; i < demographics::AGE_COUNT; i++) {
            createdPerDemographic[i] += household.sizePerAge[i];
        }
    }

    // reset per demographic count of the number initialised agents in each infection state.
    // This is used for the initial value in time series data, without having to iterate all agent data again, but may not be thread safe (i.e. probably need to change for ensembles)
    _infectedPerDemographic = {{0, 0, 0, 0, 0, 0, 0, 0, 0}};


    // ------ @todo - refactor
    // Given the household and ages are known, we can compute the workplace assignment probabilities

    // @todo - workplace enum
    constexpr std::uint8_t WORKPLACE_CHILD = 0u;
    constexpr std::uint8_t WORKPLACE_ADULT = 1u;
    constexpr std::uint8_t WORKPLACE_ELDERLY = 2u;

    // For adults, compute the likelihood they will be assigned to child or elderly network
    // @todo - prolly use a vector so we can have multiple networks instead of just 3...
    std::uint64_t n_children = createdPerDemographic[demographics::Age::AGE_0_9] + createdPerDemographic[demographics::Age::AGE_10_19];
    std::uint64_t n_elderly = createdPerDemographic[demographics::Age::AGE_70_79] + createdPerDemographic[demographics::Age::AGE_80];
    std::uint64_t n_adult = params.n_total - (n_children + n_elderly);

    // Initialise with the target number of adults in each network
    std::array<double, 3> p_adultPerWorkNetwork = {0};
    // p of each category is target number / number adults
    p_adultPerWorkNetwork[WORKPLACE_CHILD] = (params.child_network_adults * n_children) / n_adult;
    p_adultPerWorkNetwork[WORKPLACE_ELDERLY] = (params.elderly_network_adults * n_elderly) / n_adult;
    // p of being adult is then the remaining probability
    p_adultPerWorkNetwork[WORKPLACE_ADULT] = 1.0 - (p_adultPerWorkNetwork[WORKPLACE_CHILD] + p_adultPerWorkNetwork[WORKPLACE_ELDERLY]);

    // Then convert to cumulative probability with an inclusive scan
    exateppabm::util::inclusive_scan(p_adultPerWorkNetwork.begin(), p_adultPerWorkNetwork.end(), p_adultPerWorkNetwork.begin());

    // make sure the top bracket ends in a value in case of floating point / rounding >= 1.0
    p_adultPerWorkNetwork[WORKPLACE_ELDERLY] = 1.0;

    // Use a uniform distribution from 0 to 1
    std::uniform_real_distribution<float> work_network_dist(0.0f, 1.0f);

    // Prep to track how many people are in each network
    std::array<std::uint32_t, 3> peoplePerWorkplace = {{0, 0, 0}};
    // /--------

    // Populate agent data, by iterating households
    std::uint32_t personIdx = 0;
    for (std::uint32_t householdIdx = 0; householdIdx < households.size(); householdIdx++) {
        auto household = households.at(householdIdx);
        // For each individual in the household
        for (population::HouseholdSizeType householdMemberIdx = 0; householdMemberIdx < household.size; householdMemberIdx++) {
            // assert that the household structure is complete
            assert(household.size == household.agePerPerson.size());

            // Get the flamegpu person object for the individual
            auto person = pop->at(personIdx);

            // Set the individuals infection status. @todo - refactor into seir.cu?
            disease::SEIR::InfectionState infectionStatus = infected_vector.at(personIdx) ? disease::SEIR::InfectionState::Infected : disease::SEIR::InfectionState::Susceptible;
            person.setVariable<disease::SEIR::InfectionStateUnderlyingType>(exateppabm::person::v::INFECTION_STATE, infectionStatus);
            // Also set the initial infection duration. @todo - stochastic.
            float infectionStateDuration = infectionStatus == disease::SEIR::InfectionState::Infected ? params.mean_time_to_recovered: 0;
            person.setVariable<float>(exateppabm::person::v::INFECTION_STATE_DURATION, infectionStateDuration);

            // Set the individuals age and household properties
            demographics::Age age = household.agePerPerson[householdMemberIdx];
            person.setVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC, static_cast<demographics::AgeUnderlyingType>(age));
            person.setVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX, householdIdx);
            person.setVariable<std::uint8_t>(person::v::HOUSEHOLD_SIZE, household.size);

            // initialise the agents infection count
            if (infectionStatus == disease::SEIR::Infected) {
                person.setVariable<std::uint32_t>(exateppabm::person::v::INFECTION_COUNT, 1u);
                // Increment the per-age demographic initial agent count. @todo refactor elsewhere?
                _infectedPerDemographic[age]++;
            }

            // Assign the workplace based on age band - @todo refactor further
            std::uint32_t workplaceIdx = WORKPLACE_ADULT;  // default to adult workplace
            if (age == demographics::Age::AGE_0_9 || age == demographics::Age::AGE_10_19) {
                workplaceIdx = WORKPLACE_CHILD;
            } else if (age == demographics::Age::AGE_70_79 || age == demographics::Age::AGE_80) {
                workplaceIdx = WORKPLACE_ELDERLY;
            } else {
                // Randomly sample
                float r = work_network_dist(rng);
                for (std::uint32_t i = 0; i < p_adultPerWorkNetwork.size(); i++) {
                    if (r < p_adultPerWorkNetwork[i]) {
                        workplaceIdx = i;
                        break;
                    }
                }
            }

            // inc the counter per network
            peoplePerWorkplace[workplaceIdx]++;
            // Store the assigned network in the agent data structure
            person.setVariable<std::uint32_t>(person::v::WORKPLACE_IDX, workplaceIdx);

            // Increment the person index
            ++personIdx;
        }
    }

    // In a separate pass over the population, set the size of each workplace network per individual
    // @todo - refactor this once small world networks are implemented
    std::uint32_t idx = 0;
    for (auto person : *pop) {
        // get the assigned workplace
        std::uint32_t workplaceIdx = person.getVariable<std::uint32_t>(person::v::WORKPLACE_IDX);
        // Lookup the generated size
        std::uint32_t workplaceSize = peoplePerWorkplace[workplaceIdx];
        // Store in the agent's data, for faster lookup. @todo - refactor to a env property if maximum workplace count is definitely known at model definition time?
        person.setVariable<std::uint32_t>(person::v::WORKPLACE_SIZE, workplaceSize);
        // int counter
        ++idx;
    }

    // If this is a visualisation enabled build, set their x/y/z
#if defined(FLAMEGPU_VISUALISATION)
        exateppabm::visualisation::initialiseAgentPopulation(model, params, pop, static_cast<std::uint32_t>(householdSizes.size()));
#endif  // defined(FLAMEGPU_VISUALISATION)

    if (verbose) {
        // Print a summary of population creation for now.
        fmt::print("Created {} people with {} infected.\n", params.n_total, params.n_seed_infection);
        fmt::print("Households: {}\n", households.size());
        fmt::print("Demographics {{\n");
        fmt::print("   0- 9 = {}\n", createdPerDemographic[0]);
        fmt::print("  10-19 = {}\n", createdPerDemographic[1]);
        fmt::print("  20-29 = {}\n", createdPerDemographic[2]);
        fmt::print("  30-39 = {}\n", createdPerDemographic[3]);
        fmt::print("  40-49 = {}\n", createdPerDemographic[4]);
        fmt::print("  50-59 = {}\n", createdPerDemographic[5]);
        fmt::print("  60-69 = {}\n", createdPerDemographic[6]);
        fmt::print("  70-79 = {}\n", createdPerDemographic[7]);
        fmt::print("  80+   = {}\n", createdPerDemographic[8]);
        fmt::print("}}\n");
        fmt::print("Workplaces {{\n");
        fmt::print("  children: {}, adults {}\n", n_children,  peoplePerWorkplace[0] - n_children);
        fmt::print("  adults {}\n", peoplePerWorkplace[1]);
        fmt::print("  adults {}, elderly {}\n", peoplePerWorkplace[2] - n_elderly, n_elderly);
        fmt::print("}}\n");
    }

    return pop;
}

std::array<std::uint64_t, demographics::AGE_COUNT> getPerDemographicInitialInfectionCount() {
    return _infectedPerDemographic;
}

double getReferenceMeanHouseholdSize(const exateppabm::input::config& params) {
    std::vector<std::uint64_t> countPerSize = {{params.household_size_1, params.household_size_2, params.household_size_3, params.household_size_4, params.household_size_5, params.household_size_6}};
    std::uint64_t refPeople = 0u;
    std::uint64_t refHouses = 0u;
    for (std::size_t idx = 0; idx < countPerSize.size(); idx++) {
        refPeople += (idx + 1) * countPerSize[idx];
        refHouses += countPerSize[idx];
    }
    double refMeanHouseholdSize = refPeople / static_cast<double>(refHouses);
    return refMeanHouseholdSize;
}

std::vector<double> getHouseholdSizeCumulativeProbabilityVector(const exateppabm::input::config& params) {
    // Initialise vector with each config household size
    std::vector<std::uint64_t> countPerSize = {{params.household_size_1, params.household_size_2, params.household_size_3, params.household_size_4, params.household_size_5, params.household_size_6}};
    // get the sum, to find relative proportions
    std::uint64_t sumConfigHouseholdSizes = exateppabm::util::reduce(countPerSize.begin(), countPerSize.end(), 0ull);
    // Get the number of people in each household band for the reference size
    // Find the number of people that the reference household sizes can account for
    std::vector<std::uint64_t> peoplePerHouseSize = countPerSize;
    for (std::size_t idx = 0; idx < peoplePerHouseSize.size(); idx++) {
        peoplePerHouseSize[idx] = (idx + 1) * peoplePerHouseSize[idx];
    }
    std::uint64_t sumConfigPeoplePerHouseSize = exateppabm::util::reduce(peoplePerHouseSize.begin(), peoplePerHouseSize.end(), 0ull);
    double configMeanPeoplePerHouseSize = sumConfigPeoplePerHouseSize / static_cast<double>(sumConfigHouseholdSizes);

    // Build a list of household sizes, by random sampling from a uniform distribution using probabilities from the reference house size counts.
    std::vector<double> householdSizeProbability(countPerSize.size());
    for (std::size_t idx = 0; idx < householdSizeProbability.size(); idx++) {
        householdSizeProbability[idx] = countPerSize[idx] / static_cast<double>(sumConfigHouseholdSizes);
    }
    // Perform an inclusive scan to convert to cumulative probability
    exateppabm::util::inclusive_scan(householdSizeProbability.begin(), householdSizeProbability.end(), householdSizeProbability.begin());

    return householdSizeProbability;
}

std::vector<HouseholdStructure> generateHouseholdStructures(const exateppabm::input::config params, std::mt19937_64 & rng, const bool verbose) {
    /*
    @todo This method will want refactoring for realistic household generation.
    Current structure is:
    1. Get the cumulative probability distribution of house sizes based on reference data
    2. Generate household sizes randomly, using based on reference house size data
    3. For each house, generate the age per person within the household using probabilities based on global age demographic target data
    */

    // Get the vector of household size cumulative probability
    auto householdSizeProbabilityVector =  getHouseholdSizeCumulativeProbabilityVector(params);

    // Get the array of age demographic cumulative probability and reverse enum map
    auto ageDemographicProbabilities = demographics::getAgeDemographicCumulativeProbabilityArray(params);
    auto allAgeDemographics = demographics::getAllAgeDemographics();


    // Specify the rng distribution to sample from, [0, 1.0)
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // initialise a value with the number of people left to generate
    std::int64_t remainingPeople = static_cast<std::int64_t>(params.n_total);
    // estimate the number of houses, based on the cumulative probability distribution
    double refMeanHouseSize = getReferenceMeanHouseholdSize(params);
    std::uint64_t householdCountEstimate = static_cast<std::uint64_t>(std::ceil(remainingPeople / refMeanHouseSize));

    // Create a vector of house structures, and reserve enough room for the estimated number of houses
    std::vector<HouseholdStructure> households = {};
    households.reserve(householdCountEstimate);

    // create enough households for the whole population using the uniform distribution and cumulative probability vector. Ensure the last household is not too large.
    while (remainingPeople > 0) {
        double r_housesize = dist(rng);
        for (std::size_t idx = 0; idx < static_cast<HouseholdSizeType>(householdSizeProbabilityVector.size()); idx++) {
            if (r_housesize < householdSizeProbabilityVector[idx]) {
                HouseholdStructure household = {};
                household.size = static_cast<HouseholdSizeType>(idx + 1) <= remainingPeople ? static_cast<HouseholdSizeType>(idx + 1) : remainingPeople;
                household.agePerPerson.reserve(household.size);
                // Generate ages for members of the household
                for (HouseholdSizeType pidx = 0; pidx < household.size; ++pidx) {
                    float r_age = dist(rng);
                    // @todo - abstract this into a method.
                    demographics::Age age = demographics::Age::AGE_0_9;
                    for (demographics::AgeUnderlyingType i = 0; i < demographics::AGE_COUNT; i++) {
                        if (r_age < ageDemographicProbabilities[i]) {
                            age = allAgeDemographics[i];
                            break;
                        }
                    }
                    household.agePerPerson.push_back(age);
                    household.sizePerAge[static_cast<demographics::AgeUnderlyingType>(age)]++;
                }
                households.push_back(household);
                remainingPeople -= household.size;
                break;
            }
        }
    }
    // potentially shrink the vector, in case the reservation was too large
    households.shrink_to_fit();

    if (verbose) {
        // Get the count of created per house size and print it.
        std::vector<std::uint64_t> generatedHouseSizeDistribution(householdSizeProbabilityVector.size());
        for (const auto& household : households) {
            generatedHouseSizeDistribution[household.size-1]++;
        }
        fmt::print("generated households per household size (total {}) {{\n", households.size());
        for (const auto & v : generatedHouseSizeDistribution) {
            fmt::print("  {},\n", v);
        }
        fmt::print("}}\n");
        // Sum the number of people per household
        std::uint64_t sumPeoplePerHouse = std::accumulate(households.begin(), households.end(), 0ull, [](std::uint64_t tot, HouseholdStructure& h) {return tot + h.size;});
        // std::uint64_t sumPeoplePerHouse = exateppabm::util::reduce(households.begin(), households.end(), 0ull);
        // Check the mean still agrees.
        double generatedMeanPeoplePerHouseSize = sumPeoplePerHouse / static_cast<double>(households.size());
        fmt::print("generated mean household size {}\n", generatedMeanPeoplePerHouseSize);
    }
    return households;
}

}  // namespace population
}  // namespace exateppabm
