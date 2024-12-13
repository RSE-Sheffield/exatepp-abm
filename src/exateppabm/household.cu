#include "exateppabm/household.h"

#include <vector>

#include "flamegpu/flamegpu.h"
#include "fmt/core.h"
#include "exateppabm/demographics.h"
#include "exateppabm/disease.h"
#include "exateppabm/person.h"
#include "exateppabm/util.h"

namespace exateppabm {
namespace household {

/**
 * Namespace containing string constants related to the households message list
 */
namespace message_household_status {
constexpr char _NAME[] = "household_status";
__device__ constexpr char ID[] = "id";
}  // namespace message_household_status

/**
 * Agent function for person agents to emit their public information, i.e. infection status to their household
 */
FLAMEGPU_AGENT_FUNCTION(emitHouseholdStatus, flamegpu::MessageNone, flamegpu::MessageBucket) {
    // Households of 1 don't need to do any messaging, there is no one to infect
    std::uint8_t householdSize = FLAMEGPU->getVariable<std::uint8_t>(person::v::HOUSEHOLD_SIZE);
    if (householdSize > 1) {
        // output public properties to bucket message, keyed by household
        // Agent ID to avoid self messaging
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>(message_household_status::ID, FLAMEGPU->getVariable<flamegpu::id_t>(person::v::ID));

        // Household index
        // @todo - typedef or using statement for the household index type?
        std::uint32_t householdIdx = FLAMEGPU->getVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
        FLAMEGPU->message_out.setVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX, householdIdx);

        FLAMEGPU->message_out.setVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::
        INFECTION_STATE, FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE));
        FLAMEGPU->message_out.setVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC, FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC));
        // Set the message key, the house hold idx for bucket messaging @Todo
        FLAMEGPU->message_out.setKey(householdIdx);
    }
    return flamegpu::ALIVE;
}

/**
 * Very naive agent interaction for infection spread via house hold contact
 *
 * Agents iterate messages from their household members, potentially becoming infected
 *
 * @todo - refactor this somewhere else?
 * @todo - add per network behaviours?
 */
FLAMEGPU_AGENT_FUNCTION(interactHousehold, flamegpu::MessageBucket, flamegpu::MessageNone) {
    // Get my ID to avoid self messages
    const flamegpu::id_t id = FLAMEGPU->getVariable<flamegpu::id_t>(person::v::ID);  // FLAMEGPU->getID();

    // Get the probability of infection
    float p_s2e = FLAMEGPU->environment.getProperty<float>("p_interaction_susceptible_to_exposed");
    // Scale it for household interactions
    p_s2e *= FLAMEGPU->environment.getProperty<float>("relative_transmission_household");

    // Get my household index
    auto householdIdx = FLAMEGPU->getVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);

    // Get my age demographic
    auto demographic = FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);
    // Get my susceptibility modifier and modify it.
    float relativeSusceptibility = FLAMEGPU->environment.getProperty<float, demographics::AGE_COUNT>("relative_susceptibility_per_demographic", demographic);
    // Scale the probability of transmission
    p_s2e *= relativeSusceptibility;

    // Check if the current individual is susceptible to being infected
    auto infectionState = FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);

    // @todo - this will need to change for contact tracing, the message interaction will need to occur regardless.
    if (infectionState == disease::SEIR::Susceptible) {
        // Variable to store the duration of the exposed phase (if exposed)
        float stateDuration = 0.f;

        // Iterate messages from anyone within the household
        for (const auto &message : FLAMEGPU->message_in(householdIdx)) {
            // Ignore self messages (can't infect oneself)
            if (message.getVariable<flamegpu::id_t>(message_household_status::ID) != id) {
                // Ignore messages from other households
                if (message.getVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX) == householdIdx) {
                    // Check if the other agent is infected
                    if (message.getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE) == disease::SEIR::InfectionState::Infected) {
                        // Roll a dice
                        float r = FLAMEGPU->random.uniform<float>();
                        if (r < p_s2e) {
                            // I have been exposed
                            infectionState = disease::SEIR::InfectionState::Exposed;
                            // Generate how long until I am infected
                            float mean = FLAMEGPU->environment.getProperty<float>("mean_time_to_infected");
                            float sd = FLAMEGPU->environment.getProperty<float>("sd_time_to_infected");
                            stateDuration = (FLAMEGPU->random.normal<float>() * sd) + mean;
                            // @todo - for now only any exposure matters. This may want to change when quantity of exposure is important?
                            // Increment the infection counter for this individual
                            FLAMEGPU->setVariable<std::uint32_t>(person::v::INFECTION_COUNT, FLAMEGPU->getVariable<std::uint32_t>(person::v::INFECTION_COUNT) + 1);
                            break;
                        }
                    }
                }
            }
        }
        // If newly exposed, store the value in global device memory.
        if (infectionState == disease::SEIR::InfectionState::Exposed) {
            FLAMEGPU->setVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE, infectionState);
            FLAMEGPU->setVariable<float>(person::v::INFECTION_STATE_DURATION, stateDuration);
        }
    }

    return flamegpu::ALIVE;
}

void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params) {
    // Define related model environment properties
    flamegpu::EnvironmentDescription env = model.Environment();

    // define the per interaction type scale factor within households
    env.newProperty<float>("relative_transmission_household", params.relative_transmission_household);

    // Get a handle to the existing person agent type, which should have already been defined.
    flamegpu::AgentDescription agent = model.Agent(person::NAME);

    // Household network variables
    agent.newVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
    agent.newVariable<std::uint8_t>(person::v::HOUSEHOLD_SIZE);

    // Message list containing a persons current status for households (id, location, infection status)
    flamegpu::MessageBucket::Description householdStatusMessage = model.newMessage<flamegpu::MessageBucket>(message_household_status::_NAME);
    // Set the maximum bucket index to the population size. Ideally this would be household count, but that is not known at model definition time.
    // In the future this would be possible once https://github.com/FLAMEGPU/FLAMEGPU2/issues/710 is implemented
    householdStatusMessage.setUpperBound(params.n_total);
    // Add the agent id to the message.
    householdStatusMessage.newVariable<flamegpu::id_t>(message_household_status::ID);
    // Add the household index
    householdStatusMessage.newVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
    // Add a variable for the agent's infections status
    householdStatusMessage.newVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
    // Demographic?
    householdStatusMessage.newVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);

    // emit current status to the household
    flamegpu::AgentFunctionDescription emitHouseholdStatusDesc = agent.newFunction("emitHouseholdStatus", emitHouseholdStatus);
    emitHouseholdStatusDesc.setMessageOutput(message_household_status::_NAME);
    emitHouseholdStatusDesc.setMessageOutputOptional(true);
    emitHouseholdStatusDesc.setInitialState(person::states::DEFAULT);
    emitHouseholdStatusDesc.setEndState(person::states::DEFAULT);

    // Interact with other agents in the household via their messages
    flamegpu::AgentFunctionDescription interactHouseholdDesc = agent.newFunction("interactHousehold", interactHousehold);
    interactHouseholdDesc.setMessageInput(message_household_status::_NAME);
    interactHouseholdDesc.setInitialState(person::states::DEFAULT);
    interactHouseholdDesc.setEndState(person::states::DEFAULT);
}

void appendLayers(flamegpu::ModelDescription& model) {
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "emitHouseholdStatus");
    }
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "interactHousehold");
    }
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
        double r_houseSize = dist(rng);
        for (std::size_t idx = 0; idx < static_cast<HouseholdSizeType>(householdSizeProbabilityVector.size()); idx++) {
            if (r_houseSize < householdSizeProbabilityVector[idx]) {
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

}  // namespace household
}  // namespace exateppabm
