#include "exateppabm/population.h"

#include <fmt/core.h>

#include <algorithm>
#include <numeric>
#include <memory>
#include <random>
#include <vector>

#include "exateppabm/demographics.h"
#include "exateppabm/disease.h"
#include "exateppabm/person.h"
#include "exateppabm/input.h"
#include "exateppabm/util.h"

namespace exateppabm {
namespace population {

namespace {

// File-scoped array  contianing the number of infected agents per demographic from population initialisation. This needs to be made accessible to a FLAME GPU Init func due to macro environment property limitations.

std::array<std::uint64_t, demographics::AGE_COUNT> infectedPerDemographic = {};

}  // namespace

/**
 * Generate a vector of household sizes, with enough places for all individuals, using probabilities derived from reference household size distributions (in put params)
 *
 * This is not how the reference model initialises households, and this is intended to be replaced in #6 when more advanced network initialisation is implemented.
 *
 * @note max house size is 6, does not allow for 6+ as in reference data
 * @note final generated housesize will be reduced to not be too large, slighlty skewing the distribution
 */
template <typename T>
std::vector<T> generateHouseholdSizes(const exateppabm::input::config config, const bool verbose, std::mt19937_64 & rng) {
    // Initialise vector with each config household size
    std::vector<std::uint64_t> countPerSize = {{config.household_size_1, config.household_size_2, config.household_size_3, config.household_size_4, config.household_size_5, config.household_size_6}};
    // get the sum, to find relative proportions
    std::uint64_t sumConfigHouseholdSizes = std::reduce(countPerSize.begin(), countPerSize.end());
    // Get the number of people in each household band for the reference size
    // Find the number of people that the reference household sizes can account for
    std::vector<std::uint64_t> peoplePerHouseSize = countPerSize;
    for (std::size_t idx = 0; idx < peoplePerHouseSize.size(); idx++) {
        peoplePerHouseSize[idx] = (idx + 1) * peoplePerHouseSize[idx];
    }
    std::uint64_t sumConfigPeoplePerHouseSize = std::reduce(peoplePerHouseSize.begin(), peoplePerHouseSize.end());
    double configMeanPeoplePerHouseSize = sumConfigPeoplePerHouseSize / static_cast<double>(sumConfigHouseholdSizes);

    if (verbose) {
        fmt::print("reference households (total {}) {{\n", sumConfigHouseholdSizes);
        for (const auto & v : countPerSize) {
            fmt::print("  {},\n", v);
        }
        fmt::print("}}\n");
        fmt::print("reference people per household size (total {}) {{\n", sumConfigPeoplePerHouseSize);
        for (const auto & v : peoplePerHouseSize) {
            fmt::print("  {},\n", v);
        }
        fmt::print("}}\n");
        fmt::print("reference mean household size {}\n", configMeanPeoplePerHouseSize);
    }

    // Build a list of household sizes, by random sampling from a uniform distribution using probabilities from the reference house size counts.
    std::vector<double> householdSizeProbability(countPerSize.size());
    for (std::size_t idx = 0; idx < householdSizeProbability.size(); idx++) {
        householdSizeProbability[idx] = countPerSize[idx] / static_cast<double>(sumConfigHouseholdSizes);
    }
    // Perform an inclusive scan to convert to cumulative probability
    exateppabm::util::inplace_inclusive_scan(householdSizeProbability);

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::int64_t remainingPeople = static_cast<std::int64_t>(config.n_total);
    // estimate the number of houses
    std::uint64_t householdCountEstimate = static_cast<std::uint64_t>(std::ceil(remainingPeople / static_cast<double>(configMeanPeoplePerHouseSize)));
    // declare the array, and pre-allocate enough memory for the estimate
    std::vector<T> peoplePerHouse = {};
    peoplePerHouse.reserve(householdCountEstimate);

    // create enough households for the whole population using the uniform distribution and cumulative probability vector. Ensure the last household is not too large.
    while (remainingPeople > 0) {
        double r = dist(rng);
        for (std::size_t idx = 0; idx < householdSizeProbability.size(); idx++) {
            if (r < householdSizeProbability[idx]) {
                T houseSize = static_cast<T>(idx + 1) <= remainingPeople ? static_cast<T>(idx + 1) : remainingPeople;
                peoplePerHouse.push_back(houseSize);
                remainingPeople -= houseSize;
                break;
            }
        }
    }
    // potentially shrink the array, in case the reservation was too large
    peoplePerHouse.shrink_to_fit();

    if (verbose) {
        // Get the count of created per house size and print it.
        std::vector<std::uint64_t> generatedHouseSizeDistribution(countPerSize.size());
        for (const auto& houseSize : peoplePerHouse) {
            generatedHouseSizeDistribution[houseSize-1]++;
        }
        fmt::print("generated households per household size (total {}) {{\n", peoplePerHouse.size());
        for (const auto & v : generatedHouseSizeDistribution) {
            fmt::print("  {},\n", v);
        }
        fmt::print("}}\n");
        // Sum the number of people per household
        std::uint64_t sumPeoplePerHouse = std::reduce(peoplePerHouse.begin(), peoplePerHouse.end(), 0ull);
        // Check the mean still agrees.
        double generatedMeanPeoplePerHouseSize = sumPeoplePerHouse / static_cast<double>(peoplePerHouse.size());
        fmt::print("generated mean household size {}\n", generatedMeanPeoplePerHouseSize);
    }
    return peoplePerHouse;
}


std::unique_ptr<flamegpu::AgentVector> generate(flamegpu::ModelDescription& model, const exateppabm::input::config config, const bool verbose, const float env_width, const float interactionRadius) {
    fmt::print("@todo - validate config inputs when generated agents (pop size, initial infected count etc)\n");

    // @todo - assert that the requested initial population is non zero.
    auto pop = std::make_unique<flamegpu::AgentVector>(model.Agent(exateppabm::person::NAME), config.n_total);

    std::uint64_t sq_width = static_cast<std::uint64_t>(env_width);
    // float expectedNeighbours = interactionRadius * interactionRadius;
    // fmt::print("sq_width {} interactionRadius {} expectedNeighbours {}\n", sq_width, interactionRadius, expectedNeighbours);

    // seed host rng for population generation.
    // @todo - does this want to be a separate seed from the config file?
    std::mt19937_64 rng(config.rng_seed);


    // Need to initialise a fixed number of individuals as infected.
    // This not very scalable way of doing it, is to create a vector with one element per individual in the simulation, initialised to false
    // set the first n_seed_infection elements to true/1
    // Shuffle the vector,  and query at agent creation time
    // RNG sampling in-loop would be more memory efficient, but harder to guarantee that exactly enough are created. This will likely be replaced anyway, so quick and dirty is fine.
    std::vector<bool> infected_vector(config.n_total);
    std::fill(infected_vector.begin(), infected_vector.begin() + std::min(config.n_total, config.n_seed_infection), true);
    std::shuffle(infected_vector.begin(), infected_vector.end(), rng);

    // Prepare a probability matrix for selecting an age demographic for the agent based on the ratio from the configuration.
    // @todo abstract this into class/methods.
    // @todo - this hardcoded 9 is a bit grim. Maybe enums can help?
    std::uint64_t configDemographicSum = config.population_0_9 + config.population_10_19 + config.population_20_29 + config.population_30_39 + config.population_40_49 + config.population_50_59 + config.population_60_69 + config.population_70_79 + config.population_80;
    // @todo - map might be more readable than an array (incase the underlying class enum values are ever changed to be a different order?)
    std::array<float, demographics::AGE_COUNT> demographicProbabilties =  {{
        config.population_0_9 / static_cast<float>(configDemographicSum),
        config.population_10_19 / static_cast<float>(configDemographicSum),
        config.population_20_29 / static_cast<float>(configDemographicSum),
        config.population_30_39 / static_cast<float>(configDemographicSum),
        config.population_40_49 / static_cast<float>(configDemographicSum),
        config.population_50_59 / static_cast<float>(configDemographicSum),
        config.population_60_69 / static_cast<float>(configDemographicSum),
        config.population_70_79 / static_cast<float>(configDemographicSum),
        config.population_80 / static_cast<float>(configDemographicSum)
    }};
    // Perform an inclusive scan to convert to cumulative probability
    // Using a local method which supports inclusive scans in old libstc++
    exateppabm::util::inplace_inclusive_scan(demographicProbabilties);
    std::array<demographics::Age, demographics::AGE_COUNT> allDemographics = {{
        demographics::Age::AGE_0_9,
        demographics::Age::AGE_10_19,
        demographics::Age::AGE_20_29,
        demographics::Age::AGE_30_39,
        demographics::Age::AGE_40_49,
        demographics::Age::AGE_50_59,
        demographics::Age::AGE_60_69,
        demographics::Age::AGE_70_79,
        demographics::Age::AGE_80
    }};

    // per demo total is not an output in time series.
    // Alternately, we need to initialise the exact number of each age band, not RNG, and just scale it down accordingly. Will look at in "realistic" population generation
    std::array<std::uint64_t, demographics::AGE_COUNT> createdPerDemographic = {{0, 0, 0, 0, 0, 0, 0, 0, 0}};
    // reset per demographic count of the number initialised agents in each infection state.
    infectedPerDemographic = {{0, 0, 0, 0, 0, 0, 0, 0, 0}};

    std::uniform_real_distribution<float> demo_dist(0.0f, 1.0f);

    // Generate a vector containing the number of individuals for each household
    auto householdSizes = generateHouseholdSizes<std::uint8_t>(config, verbose, rng);
    auto householdIndexPerPerson = std::vector<std::uint32_t>(config.n_total, 0);
    // Generate a vector for each person to be created, which contains the index of the house it will be assigned to.
    std::size_t hippOffset = 0;
    for (std::size_t houseIdx = 0; houseIdx < householdSizes.size(); houseIdx++) {
        auto houseSize = householdSizes[houseIdx];
        std::fill_n(householdIndexPerPerson.begin() + hippOffset, houseSize, houseIdx);
        hippOffset += houseSize;
    }
    // @todo - Shuffle the vector? Not strictly neccessary as house sizes are randomized?

    unsigned idx = 0;
    for (auto person : *pop) {
        // Infections status. @todo - refactor into seir.cu?
        disease::SEIR::InfectionState infectionStatus = infected_vector.at(idx) ? disease::SEIR::InfectionState::Infected : disease::SEIR::InfectionState::Susceptible;
        person.setVariable<disease::SEIR::InfectionStateUnderlyingType>(exateppabm::person::v::INFECTION_STATE, infectionStatus);
        // Also set the initial infection duration. @todo - stochastic.
        float infectionStateDuration = infectionStatus == disease::SEIR::InfectionState::Infected ? config.mean_time_to_recovered: 0;
        person.setVariable<float>(exateppabm::person::v::INFECTION_STATE_DURATION, infectionStateDuration);

        // initialise the agents infection count
        if (infectionStatus == disease::SEIR::Infected) {
            person.setVariable<std::uint32_t>(exateppabm::person::v::INFECTION_COUNT, 1);
        }

        // Demographic
        // @todo - this is a bit grim, enum class aren't as nice as hoped.
        float demo_random = demo_dist(rng);
        // @todo - abstract this into a method.
        demographics::Age demo = demographics::Age::AGE_0_9;
        for (demographics::AgeUnderlyingType i = 0; i < demographics::AGE_COUNT; i++) {
            if (demo_random < demographicProbabilties[i]) {
                demo = allDemographics[i];
                createdPerDemographic[i]++;
                if (infectionStatus == disease::SEIR::Infected) {
                    infectedPerDemographic[i]++;
                }
                break;
            }
        }
        person.setVariable<demographics::AgeUnderlyingType>(exateppabm::person::v::AGE_DEMOGRAPHIC, demo);

        // Household assignment
        std::uint32_t householdIdx = householdIndexPerPerson[idx];
        person.setVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX, householdIdx);
        std::uint8_t householdSize = householdSizes[householdIdx];
        person.setVariable<std::uint8_t>(person::v::HOUSEHOLD_SIZE, householdSize);

        // Work/occupation networks can only be set once all age demographics are known, so done in a subsequent loop. @todo this method needs refactoring eventually.

        // Location in 3D space (temp/vis)
        unsigned row = idx / sq_width;
        unsigned col = idx % sq_width;
        person.setVariable<float>(exateppabm::person::v::x, col);  // @todo temp
        person.setVariable<float>(exateppabm::person::v::y, row);  // @todo -temp
        person.setVariable<float>(exateppabm::person::v::z, 0);  // @todo -temp

        // Inc counter
        ++idx;
    }

    // Workplace per individual can now be set, as the total number of young and old people must be known to compute the distribution for adults.
    // This is likely to be refactored.

    // For adults, compute the likelihood they will be assigned to child or elderly network
    // @todo - prolly use a vector so we can have multiple networks instead of just 3...
    std::uint64_t n_children = createdPerDemographic[demographics::Age::AGE_0_9] + createdPerDemographic[demographics::Age::AGE_10_19];
    std::uint64_t n_elderly = createdPerDemographic[demographics::Age::AGE_70_79] + createdPerDemographic[demographics::Age::AGE_80];
    std::uint64_t n_adult = config.n_total - (n_children + n_elderly);

    // Initialise with the target number of adults in each network
    std::array<double, 3> p_adultPerWorkNetwork = {0};
    // p of each category is target number / number adults
    p_adultPerWorkNetwork[0] = (config.child_network_adults * n_children) / n_adult;
    p_adultPerWorkNetwork[2] = (config.elderly_network_adults * n_elderly) / n_adult;
    // p of being adult is then the remaining probabilty
    p_adultPerWorkNetwork[1] = 1.0 - (p_adultPerWorkNetwork[0] + p_adultPerWorkNetwork[2]);

    // Then convert to cumulative probability with an inclusive scan
    exateppabm::util::inplace_inclusive_scan(p_adultPerWorkNetwork);

    // make sure the top bracket ends in >=1.0
    p_adultPerWorkNetwork[2] = 1.0;
    // fmt::print("p work adult: {} {} {}\n",p_adultPerWorkNetwork[0], p_adultPerWorkNetwork[1],p_adultPerWorkNetwork[2]);

    // Use a uniform distribution from 0 to 1
    std::uniform_real_distribution<float> work_network_dist(0.0f, 1.0f);


    // Prep to track how many people are in each network
    std::array<std::uint32_t, 3> peoplePerWorkplace = {{0, 0, 0}};

    // @todo - add an enum.

    idx = 0;
    for (auto person : *pop) {
        // Assign the workplace based on age band
        std::uint32_t workplaceIdx = 1;
        demographics::Age age = static_cast<demographics::Age>(person.getVariable<demographics::AgeUnderlyingType>(exateppabm::person::v::AGE_DEMOGRAPHIC));

        if (age == demographics::Age::AGE_0_9 || age == demographics::Age::AGE_10_19) {
            workplaceIdx = 0;  // @todo enum?
        } else if (age == demographics::Age::AGE_70_79 || age == demographics::Age::AGE_80) {
            workplaceIdx = 2;  // @todo enum?
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
        ++idx;
    }

    // Finally we can assign the size of each network in another separate pass. Would be nice if we could do this in less passes...

    idx = 0;
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

    if (verbose) {
        // Print a summary of population creation for now.
        fmt::print("Created {} people with {} infected.\n", config.n_total, config.n_seed_infection);
        fmt::print("Households: {}\n", householdSizes.size());
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
    return infectedPerDemographic;
}

}  // namespace population
}  // namespace exateppabm
