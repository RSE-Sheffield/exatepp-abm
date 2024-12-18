#include "exateppabm/random_interactions.h"

#include <vector>

#include "flamegpu/flamegpu.h"
#include "exateppabm/demographics.h"
#include "exateppabm/disease.h"
#include "exateppabm/person.h"
#include "exateppabm/population.h"  // @todo - replace with workpalce when abstracted

namespace exateppabm {
namespace random_interactions {

/**
 * Namespace containing string constants related to messages lists within random interactions
 * Use of __device__ constexpr char [] requires CUDA >= 11.4
 */
namespace message_random_network_status {
    constexpr char _NAME[] = "random_network_status";
}  // namespace message_random_network_status

/**
 * Update the per-day random daily interaction network
 *
 * Individuals have a number of daily interactions, which is the same for the duration of th simulation.
 *
 * In serial on the host, generate a vector of interactions
 *
 * @todo - more than one random interaction per day
 * @todo - prevent duplicate interactions
 * @todo - more performant CPU implementation
 * @todo - GPU implementation of this (stable matching submodel?)
 */
FLAMEGPU_HOST_FUNCTION(updateRandomDailyNetworkIndices) {
    // Get the current population of person agents
    auto personAgent = FLAMEGPU->agent(exateppabm::person::NAME, exateppabm::person::states::DEFAULT);
    flamegpu::DeviceAgentVector population = personAgent.getPopulationData();

    // @todo implement the following
    // Get the sum of the per-agent random interaction count, so we can reserve a large enough vector
    std::uint64_t randomInteractionCountSum = FLAMEGPU->environment.getProperty<std::uint64_t>("RANDOM_INTERACTION_COUNT_SUM");

    // If there are none, return.
    if (randomInteractionCountSum == 0) {
        return;
    }

    // Build a vector of interactions, initialised to contain the id of each agent as many times as they require
    // Number is fixed for the simulation, so only allocate and initialise once via a method-static variable
    static std::vector<flamegpu::id_t> randomInteractionIdVector;
    // If empty, initialise to contain each agent's ID the required number of times.
    // This will only trigger a device to host copy for the first pass.
    // This could be optimised away, with a namespace scoped static and initialised during initial host agent population, at the cost of readability
    if (randomInteractionIdVector.empty()) {
        randomInteractionIdVector.resize(randomInteractionCountSum);
        size_t first = 0;
        size_t last = 0;
        for (const auto &person : population) {
            flamegpu::id_t id = person.getVariable<flamegpu::id_t>(person::v::ID);
            std::uint32_t count = person.getVariable<std::uint32_t>(person::v::RANDOM_INTERACTION_COUNT_TARGET);
            last = first + count;
            if (last > randomInteractionCountSum) {
                throw std::runtime_error("Too many random interactions being generated. @todo better error");
            }
            std::fill(randomInteractionIdVector.begin() + first, randomInteractionIdVector.begin() + last, id);
            first = last;
        }
    }

    // Shuffle the vector of agent id's.
    // @todo - adjust how this RNG is seeded, this is far from ideal.
    auto rng = std::mt19937_64(FLAMEGPU->random.uniform<double>());
    std::shuffle(randomInteractionIdVector.begin(), randomInteractionIdVector.end(), rng);

    // Update agent data containing today's random interactions
    // This will trigger Host to Device copies
    // Iterate the pairs of ID which form the interactions, updating both agents' data for each interaction.
    // @todo - avoid self-interactions and duplicate interactions.
    // The number of interaction pairs is the total number of id's divided by 2, and rounded down.
    std::uint64_t interactionCount = static_cast<std::uint64_t>(std::floor(randomInteractionCountSum / 2.0));
    for (std::uint64_t interactionIdx = 0; interactionIdx < interactionCount; ++interactionIdx) {
        // Get the indices within the vector of agent id's. An mdspan would be nice if not c++17
        size_t aIndex = (interactionIdx * 2);
        size_t bIndex = aIndex + 1;

        // Get the agent ID's
        flamegpu::id_t aID = randomInteractionIdVector[aIndex];
        flamegpu::id_t bID = randomInteractionIdVector[bIndex];

        // Assuming that agents are in-oder (which is not guaranteed!)
        // @todo - this validation could be done once per iteration, not once per interaction?
        // @todo switch to a sparse data structure, indexed by agent ID? Not ideal for mem access, but simpler and less validation required.

        // Get a handle to each agent from the population vector.
        // Agents ID's are 1 indexed
        auto aAgent = population[aID - 1];
        auto bAgent = population[bID - 1];
        // Raise an exception if the agents are out of order. This should not be the case as agent birth, death and sorting should not be included in this model.
        if (aAgent.getVariable<flamegpu::id_t>(person::v::ID) != aID || bAgent.getVariable<flamegpu::id_t>(person::v::ID) != bID) {
            throw std::runtime_error("Agent ID does not match expected agent ID in updateRandomDailyNetworkIndices. @todo");
        }

        // @todo - avoid self-interactions by looking ahead and swapping?
        // @todo - avoid repeated interactions by looking ahead and swapping?

        // Add the interaction to the list of interactions agent a
        std::uint32_t aInteractionIdx = aAgent.getVariable<std::uint32_t>(person::v::RANDOM_INTERACTION_COUNT);
        // aAgent.setVariable<flamegpu::id_t, person::MAX_RANDOM_DAILY_INTERACTIONS>(person::v::RANDOM_INTERACTION_PARTNERS, aInteractionIdx, bID);
        aAgent.setVariable<flamegpu::id_t>(person::v::RANDOM_INTERACTION_PARTNERS, aInteractionIdx, bID);
        aAgent.setVariable<std::uint32_t>(person::v::RANDOM_INTERACTION_COUNT, aInteractionIdx + 1);


        // Add the interaction to the list of interactions agent b
        std::uint32_t bInteractionIdx = bAgent.getVariable<std::uint32_t>(person::v::RANDOM_INTERACTION_COUNT);
        // bAgent.setVariable<flamegpu::id_t, person::MAX_RANDOM_DAILY_INTERACTIONS>(person::v::RANDOM_INTERACTION_PARTNERS, bInteractionIdx, aID);
        bAgent.setVariable<flamegpu::id_t>(person::v::RANDOM_INTERACTION_PARTNERS, bInteractionIdx, aID);
        bAgent.setVariable<std::uint32_t>(person::v::RANDOM_INTERACTION_COUNT, bInteractionIdx + 1);
    }

    // setVariable<T, LEN>(name, idx, value) in flamegpu 2.0.0-rc.2 and earlier contains a bug where the single element setting variant does not trigger data synchronisation on the end of the host function
    // This has been fixed by https://github.com/FLAMEGPU/FLAMEGPU2/pull/1266, which will be part of flamegpu 2.0.0-rc.3
    // As pre release versions are strings, this is a more complex if statement than it would ideally be
    // Can macro this out if FLAME GPU is not 2.0.0
#if defined(FLAMEGPU_VERSION) && FLAMEGPU_VERSION == 2000000
    // then at runtime we can only trigger the workaround for flamegpu2 versions rc2 and before
    if (strcmp(flamegpu::VERSION_PRERELEASE, "rc.2") == 0
        || strcmp(flamegpu::VERSION_PRERELEASE, "rc.1") == 0
        || strcmp(flamegpu::VERSION_PRERELEASE, "rc") == 0
        || strcmp(flamegpu::VERSION_PRERELEASE, "alpha.2") == 0
        || strcmp(flamegpu::VERSION_PRERELEASE, "alpha.1") == 0
        || strcmp(flamegpu::VERSION_PRERELEASE, "alpha") == 0) {
        for (auto person : population) {
            person.setVariable<flamegpu::id_t, person::MAX_RANDOM_DAILY_INTERACTIONS>(person::v::RANDOM_INTERACTION_PARTNERS, person.getVariable<flamegpu::id_t, person::MAX_RANDOM_DAILY_INTERACTIONS>(person::v::RANDOM_INTERACTION_PARTNERS));
        }
    }
#endif
}

/**
 * Agent function for person agents to emit their public information
 * i.e. infection status, for random daily network colleagues.
 *
 */
FLAMEGPU_AGENT_FUNCTION(emitRandomDailyNetworkStatus, flamegpu::MessageNone, flamegpu::MessageArray) {
    // output public properties to array message, keyed by the agent ID so no need to include the ID.

    FLAMEGPU->message_out.setVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::
    INFECTION_STATE, FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE));
    FLAMEGPU->message_out.setVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC, FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC));

    // Set the message array message index to the agent's id.
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<flamegpu::id_t>(person::v::ID));
    return flamegpu::ALIVE;
}

/**
 * Very naive agent interaction for infection spread via random daily network contact
 *
 * Agents iterate messages from others within their random daily network, potentially becoming infected
 *
 * @todo - refactor this somewhere else?
 * @todo - add per network behaviours?
 * @todo - leverage a FLAME GPU 2 network structure, when they are mutable per day.
 */
FLAMEGPU_AGENT_FUNCTION(interactRandomDailyNetwork, flamegpu::MessageArray, flamegpu::MessageNone) {
    // Get my ID to avoid self messages
    const flamegpu::id_t id = FLAMEGPU->getVariable<flamegpu::id_t>(person::v::ID);

    // Get the probability of infection
    float p_s2e = FLAMEGPU->environment.getProperty<float>("p_interaction_susceptible_to_exposed");
     // Scale it for random daily interactions
    p_s2e *= FLAMEGPU->environment.getProperty<float>("relative_transmission_random");

    // Get my age demographic
    auto demographic = FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);
    // Get my susceptibility modifier and modify it.
    float relativeSusceptibility = FLAMEGPU->environment.getProperty<float, demographics::AGE_COUNT>("relative_susceptibility_per_demographic", demographic);
    // Scale the probability of transmission
    p_s2e *= relativeSusceptibility;

    // Check if the current individual is susceptible to being infected
    auto infectionState = disease::SEIR::getCurrentInfectionStatus(FLAMEGPU);

    // Only check interactions from this individual if they are susceptible. @todo - this will need to change for contact tracing.
    if (infectionState == disease::SEIR::Susceptible) {
        // Bool to track if individual newly exposed - used to move expensive operations outside the message iteration loop.
        bool newlyExposed = false;
        // For each interaction this agent is set to perform
        const std::uint32_t randomInteractionCount = FLAMEGPU->getVariable<std::uint32_t>(person::v::RANDOM_INTERACTION_COUNT);
        for (std::uint32_t randomInteractionIdx = 0; randomInteractionIdx < randomInteractionCount; ++randomInteractionIdx) {
            // Get the (next) ID of the interaction partner.
            flamegpu::id_t otherID = FLAMEGPU->getVariable<flamegpu::id_t, person::MAX_RANDOM_DAILY_INTERACTIONS>(person::v::RANDOM_INTERACTION_PARTNERS, randomInteractionIdx);
            // if the ID is not self and not unset
            if (otherID != id && otherID != flamegpu::ID_NOT_SET) {
                // Get the message handle
                const auto &message = FLAMEGPU->message_in.at(otherID);
                // Check if the other agent is infected
                if (message.getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE) == disease::SEIR::InfectionState::Infected) {
                    // Roll a dice
                    float r = FLAMEGPU->random.uniform<float>();
                    if (r < p_s2e) {
                        // set a flag indicating that the individual has been exposed in this message iteration loop
                        newlyExposed = true;
                        // break out of the loop over today's random interactions - can only be exposed once
                        break;
                    }
                }
            }
        }
        // If newly exposed, update agent data and generate new seir state information. This is done outside the message iteration loop to be more GPU-shaped.
        if (newlyExposed) {
            // Transition from susceptible to exposed in SEIR
            disease::SEIR::susceptibleToExposed(FLAMEGPU, infectionState);
        }
    }

    // reset the agent's number of interactions to 0 in advance of the next day.
    // This is expensive, but the D2H copy would get triggered anyway if attempting to update on the host anyway.
    // @todo - a more performant way to do the random daily interaction network would be good.
    FLAMEGPU->setVariable<std::uint32_t>(person::v::RANDOM_INTERACTION_COUNT, 0u);

    return flamegpu::ALIVE;
}


void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params) {
    // Define related model environment properties
    flamegpu::EnvironmentDescription env = model.Environment();

    // Environmental property containing the relative transmission scale factor for random interactions
    env.newProperty<float>("relative_transmission_random", params.relative_transmission_random);

    // Add an environmental variable containing the sum of each agents target number of random interactions.
    // This is not the number of pair-wise interactions, but the number each individual would like.
    // there are cases where this value would be impossible to achieve for a given population.
    // I.e. if there are 2 agents, with targets of [2, 1]. This would be a target of 3, but only 1 interaction is possible
    env.newProperty<std::uint64_t>("RANDOM_INTERACTION_COUNT_SUM", 0u);

    // Get a handle to the existing person agent type, which should have already been defined.
    flamegpu::AgentDescription agent = model.Agent(person::NAME);

    // Define person agent variables related to the random interactions

    // The target number of random interactions for this individual every day
    agent.newVariable<std::uint32_t>(person::v::RANDOM_INTERACTION_COUNT_TARGET, 0u);

    // The number of random interactions this individual has for the current day
    agent.newVariable<std::uint32_t>(person::v::RANDOM_INTERACTION_COUNT, 0u);

    // An array with a compile-time defined size for neighbors to interact with
    agent.newVariable<flamegpu::id_t, person::MAX_RANDOM_DAILY_INTERACTIONS>(person::v::RANDOM_INTERACTION_PARTNERS, {flamegpu::ID_NOT_SET});

    // Message list containing a persons current status for random,daily interaction (id, location, infection status)
    // Uses an array message with 1 per agent
    flamegpu::MessageArray::Description randomNetworkStatusMessage = model.newMessage<flamegpu::MessageArray>(message_random_network_status::_NAME);
    // ID's are 1 indexed (0 is unset) so use + 1.
    // For ensembles this will need to be the largest n_total until https://github.com/FLAMEGPU/FLAMEGPU2/issues/710 is implemented
    randomNetworkStatusMessage.setLength(params.n_total + 1);
    // No need to add the agent ID to the message, it's the same as the array index
    // Add a variable for the agent's infections status
    randomNetworkStatusMessage.newVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
    // Agent's demographic
    randomNetworkStatusMessage.newVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);

    // Define host and agent functions

    // Host function to generate todays random interaction list.
    // @todo - a GPU implementation will be needed for model scaling.
    // @todo - or we could construct the next days list async during the previous day & double buffer?
    flamegpu::HostFunctionDescription("updateRandomDailyNetworkIndices", updateRandomDailyNetworkIndices);

    // emit current status for random interactions
    flamegpu::AgentFunctionDescription emitRandomDailyNetworkStatusDesc = agent.newFunction("emitRandomDailyNetworkStatus", emitRandomDailyNetworkStatus);
    emitRandomDailyNetworkStatusDesc.setMessageOutput(message_random_network_status::_NAME);
    emitRandomDailyNetworkStatusDesc.setInitialState(person::states::DEFAULT);
    emitRandomDailyNetworkStatusDesc.setEndState(person::states::DEFAULT);

    // Interact with other agents in the random interactions
    flamegpu::AgentFunctionDescription interactRandomDailyNetworkDesc = agent.newFunction("interactRandomDailyNetwork", interactRandomDailyNetwork);
    interactRandomDailyNetworkDesc.setMessageInput(message_random_network_status::_NAME);
    interactRandomDailyNetworkDesc.setInitialState(person::states::DEFAULT);
    interactRandomDailyNetworkDesc.setEndState(person::states::DEFAULT);
}

void appendLayers(flamegpu::ModelDescription& model) {
    {
        auto layer = model.newLayer();
        layer.addHostFunction(updateRandomDailyNetworkIndices);
    }
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "emitRandomDailyNetworkStatus");
    }
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "interactRandomDailyNetwork");
    }
}

}  // namespace random_interactions
}  // namespace exateppabm
