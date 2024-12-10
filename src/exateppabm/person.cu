#include "exateppabm/person.h"

#include <fmt/core.h>

#include <vector>

#include "exateppabm/disease/SEIR.h"
#include "exateppabm/demographics.h"
#include "exateppabm/population.h"  // @todo - replace with workpalce when abstracted


namespace exateppabm {
namespace person {

/**
 * Agent function for person agents to emit their public information, i.e. infection status to their household
 */
FLAMEGPU_AGENT_FUNCTION(emitHouseholdStatus, flamegpu::MessageNone, flamegpu::MessageBucket) {
    // Households of 1 don't need to do any messaging, there is no one to infect
    std::uint8_t householdSize = FLAMEGPU->getVariable<std::uint8_t>(v::HOUSEHOLD_SIZE);
    if (householdSize > 1) {
        // output public properties to bucket message, keyed by household
        // Agent ID to avoid self messaging
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>(person::message::household_status::ID, FLAMEGPU->getID());

        // Household index
        // @todo - typedef or using statement for the household index type?
        std::uint32_t householdIdx = FLAMEGPU->getVariable<std::uint32_t>(v::HOUSEHOLD_IDX);
        FLAMEGPU->message_out.setVariable<std::uint32_t>(v::HOUSEHOLD_IDX, householdIdx);

        FLAMEGPU->message_out.setVariable<disease::SEIR::InfectionStateUnderlyingType>(v::
        INFECTION_STATE, FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE));
        FLAMEGPU->message_out.setVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC, FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC));
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
    const flamegpu::id_t id = FLAMEGPU->getID();

    // Get the probability of infection
    float p_s2e = FLAMEGPU->environment.getProperty<float>("p_interaction_susceptible_to_exposed");
    // Scale it for household interactions
    p_s2e *= FLAMEGPU->environment.getProperty<float>("relative_transmission_household");

    // Get my household index
    auto householdIdx = FLAMEGPU->getVariable<std::uint32_t>(v::HOUSEHOLD_IDX);

    // Get my age demographic
    auto demographic = FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC);
    // Get my susceptibility modifier and modify it.
    float relativeSusceptibility = FLAMEGPU->environment.getProperty<float, demographics::AGE_COUNT>("relative_susceptibility_per_demographic", demographic);
    // Scale the probability of transmission
    p_s2e *= relativeSusceptibility;

    // Check if the current individual is susceptible to being infected
    auto infectionState = FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE);

    // @todo - this will need to change for contact tracing, the message interaction will need to occur regardless.
    if (infectionState == disease::SEIR::Susceptible) {
        // Variable to store the duration of the exposed phase (if exposed)
        float stateDuration = 0.f;

        // Iterate messages from anyone within the household
        for (const auto &message : FLAMEGPU->message_in(householdIdx)) {
            // Ignore self messages (can't infect oneself)
            if (message.getVariable<flamegpu::id_t>(message::household_status::ID) != id) {
                // Ignore messages from other households
                if (message.getVariable<std::uint32_t>(v::HOUSEHOLD_IDX) == householdIdx) {
                    // Check if the other agent is infected
                    if (message.getVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE) == disease::SEIR::InfectionState::Infected) {
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
                            FLAMEGPU->setVariable<std::uint32_t>(v::INFECTION_COUNT, FLAMEGPU->getVariable<std::uint32_t>(v::INFECTION_COUNT) + 1);
                            break;
                        }
                    }
                }
            }
        }
        // If newly exposed, store the value in global device memory.
        if (infectionState == disease::SEIR::InfectionState::Exposed) {
            FLAMEGPU->setVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE, infectionState);
            FLAMEGPU->setVariable<float>(person::v::INFECTION_STATE_DURATION, stateDuration);
        }
    }

    return flamegpu::ALIVE;
}



/**
 * Agent function for person agents to emit their public information, i.e. infection status, to their workplace colleagues
 */
FLAMEGPU_AGENT_FUNCTION(emitWorkplaceStatus, flamegpu::MessageNone, flamegpu::MessageBucket) {
    // workplaces of 1 don't need to do any messaging, there is no one to infect
    std::uint32_t workplaceSize = FLAMEGPU->getVariable<std::uint32_t>(v::WORKPLACE_SIZE);
    if (workplaceSize > 1) {
        // output public properties to bucket message, keyed by workplace
        // Agent ID to avoid self messaging
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>(person::message::workplace_status::ID, FLAMEGPU->getID());

        // workplace index
        // @todo - typedef or using statement for the workplace index type?
        std::uint32_t workplaceIdx = FLAMEGPU->getVariable<std::uint32_t>(v::WORKPLACE_IDX);
        FLAMEGPU->message_out.setVariable<std::uint32_t>(v::WORKPLACE_IDX, workplaceIdx);

        FLAMEGPU->message_out.setVariable<disease::SEIR::InfectionStateUnderlyingType>(v::
        INFECTION_STATE, FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE));
        FLAMEGPU->message_out.setVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC, FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC));

        // Set the message key, the house hold idx for bucket messaging @Todo
        FLAMEGPU->message_out.setKey(workplaceIdx);
    }
    return flamegpu::ALIVE;
}

/**
 * Very naive agent interaction for infection spread via workplace contact
 *
 * Agents iterate messages from their workplace members, potentially becoming infected
 *
 * @todo - refactor this somewhere else?
 * @todo - add per network behaviours?
 */
FLAMEGPU_AGENT_FUNCTION(interactWorkplace, flamegpu::MessageBucket, flamegpu::MessageNone) {
    // Get my ID to avoid self messages
    const flamegpu::id_t id = FLAMEGPU->getID();

    // Get the probability of interaction within the workplace
    float p_daily_fraction_work = FLAMEGPU->environment.getProperty<float>("daily_fraction_work");

    // Get the probability of infection
    float p_s2e = FLAMEGPU->environment.getProperty<float>("p_interaction_susceptible_to_exposed");
     // Scale it for workplace interactions
    p_s2e *= FLAMEGPU->environment.getProperty<float>("relative_transmission_occupation");

    // Get my workplace/network index
    auto workplaceIdx = FLAMEGPU->getVariable<std::uint32_t>(v::WORKPLACE_IDX);

    // Get my age demographic
    auto demographic = FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC);
    // Get my susceptibility modifier and modify it.
    float relativeSusceptibility = FLAMEGPU->environment.getProperty<float, demographics::AGE_COUNT>("relative_susceptibility_per_demographic", demographic);
    // Scale the probability of transmission
    p_s2e *= relativeSusceptibility;

    // Check if the current individual is susceptible to being infected
    auto infectionState = FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE);

// @todo - only interact with some others

    // @todo - this will need to change for contact tracing, the message interaction will need to occur regardless.
    if (infectionState == disease::SEIR::Susceptible) {
        // Variable to store the duration of the exposed phase (if exposed)
        float stateDuration = 0.f;

        // Iterate messages from anyone within my workplace
        for (const auto &message : FLAMEGPU->message_in(workplaceIdx)) {
            // Ignore self messages (can't infect oneself)
            if (message.getVariable<flamegpu::id_t>(message::workplace_status::ID) != id) {
                // Ignore messages from other workplaces
                if (message.getVariable<std::uint32_t>(v::WORKPLACE_IDX) == workplaceIdx) {
                    // roll a dice to determine if this interaction should occur this day
                    if (FLAMEGPU->random.uniform<float>() < p_daily_fraction_work) {
                        // Check if the other agent is infected
                        if (message.getVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE) == disease::SEIR::InfectionState::Infected) {
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
                                FLAMEGPU->setVariable<std::uint32_t>(v::INFECTION_COUNT, FLAMEGPU->getVariable<std::uint32_t>(v::INFECTION_COUNT) + 1);
                                break;
                            }
                        }
                    }
                }
            }
        }
        // If newly exposed, store the value in global device memory.
        if (infectionState == disease::SEIR::InfectionState::Exposed) {
            FLAMEGPU->setVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE, infectionState);
            FLAMEGPU->setVariable<float>(person::v::INFECTION_STATE_DURATION, stateDuration);
        }
    }

    return flamegpu::ALIVE;
}


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
            flamegpu::id_t id = person.getID();
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
        if (aAgent.getID() != aID || bAgent.getID() != bID) {
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
 * Agent function for person agents to emit their public information, i.e. infection status, for random daily network colleagues. This is put into a bucket key'd by the agent's ID.
 *
 * @note the bucket message is from 1...N, rather than 0 to match ID index.
 */
FLAMEGPU_AGENT_FUNCTION(emitRandomDailyNetworkStatus, flamegpu::MessageNone, flamegpu::MessageArray) {
    // output public properties to array message
    // FLAMEGPU->message_out.setVariable<flamegpu::id_t>(person::message::workplace_status::ID, FLAMEGPU->getID());

    FLAMEGPU->message_out.setVariable<disease::SEIR::InfectionStateUnderlyingType>(v::
    INFECTION_STATE, FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE));
    FLAMEGPU->message_out.setVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC, FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC));

    // Set the message array message index to the agent's id.
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getID());
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
    const flamegpu::id_t id = FLAMEGPU->getID();

    // Get the probability of infection
    float p_s2e = FLAMEGPU->environment.getProperty<float>("p_interaction_susceptible_to_exposed");
     // Scale it for random daily interactions
    p_s2e *= FLAMEGPU->environment.getProperty<float>("relative_transmission_random");

    // Get my age demographic
    auto demographic = FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC);
    // Get my susceptibility modifier and modify it.
    float relativeSusceptibility = FLAMEGPU->environment.getProperty<float, demographics::AGE_COUNT>("relative_susceptibility_per_demographic", demographic);
    // Scale the probability of transmission
    p_s2e *= relativeSusceptibility;

    // Check if the current individual is susceptible to being infected
    auto infectionState = FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE);

    // @todo - this will need to change for contact tracing, the message interaction will need to occur regardless.
    if (infectionState == disease::SEIR::Susceptible) {
        // Variable to store the duration of the exposed phase (if exposed)
        float stateDuration = 0.f;

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
                if (message.getVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE) == disease::SEIR::InfectionState::Infected) {
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
                        FLAMEGPU->setVariable<std::uint32_t>(v::INFECTION_COUNT, FLAMEGPU->getVariable<std::uint32_t>(v::INFECTION_COUNT) + 1);
                        break;
                    }
                }
            }
        }
        // If newly exposed, store the value in global device memory.
        if (infectionState == disease::SEIR::InfectionState::Exposed) {
            FLAMEGPU->setVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE, infectionState);
            FLAMEGPU->setVariable<float>(person::v::INFECTION_STATE_DURATION, stateDuration);
        }
    }

    // reset the agent's number of interactions to 0 in advance of the next day.
    // This is expensive, but the D2H copy would get triggered anyway if attempting to update on the host anyway.
    // @todo - a more performant way to do the random daily interaction network would be good.
    FLAMEGPU->setVariable<std::uint32_t>(person::v::RANDOM_INTERACTION_COUNT, 0u);

    return flamegpu::ALIVE;
}

void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params) {
    // Define related model environment properties (@todo - abstract these somewhere more appropriate at a later date)
    flamegpu::EnvironmentDescription env = model.Environment();

    // @todo this should probably be refactored elsewhere (although used in this file currently)
    // Define the base probability of being exposed if interacting with an infected individual
    env.newProperty<float>("p_interaction_susceptible_to_exposed", params.p_interaction_susceptible_to_exposed);
    // define the per interaction type scale factors as env var
    env.newProperty<float>("relative_transmission_household", params.relative_transmission_household);
    env.newProperty<float>("relative_transmission_occupation", params.relative_transmission_occupation);
    env.newProperty<float>("relative_transmission_random", params.relative_transmission_random);

    // env var for the fraction of people in the same work network to interact with
    env.newProperty<float>("daily_fraction_work", params.daily_fraction_work);

    // Define the agent type
    flamegpu::AgentDescription agent = model.newAgent(person::NAME);

    // Define states
    agent.newState(person::states::DEFAULT);

    // Define variables
    // disease related variables
    // @todo - define this in disease/ call a disease::SEIR::define_person() like method?
    agent.newVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE, disease::SEIR::Susceptible);
    // Timestep/day of last state change
    agent.newVariable<std::uint32_t>(person::v::INFECTION_STATE_CHANGE_DAY, 0);
    // Time until next state change? Defaults to the simulation duration + 1.
    agent.newVariable<float>(person::v::INFECTION_STATE_DURATION, params.duration + 1);

    // Integer count for the number of times infected, defaults to 0
    agent.newVariable<std::uint32_t>(person::v::INFECTION_COUNT, 0u);

    // age demographic
    // @todo make this an enum, and update uses of it, but flame's templating disagrees?
    agent.newVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC, exateppabm::demographics::Age::AGE_0_9);

    // Household network variables. @todo - refactor to a separate network location?
    agent.newVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
    agent.newVariable<std::uint8_t>(person::v::HOUSEHOLD_SIZE);

    // Workplace environment directed graph
    // This single graph contains workplace information for all individuals, and is essentially 5 unconnected sub graphs.
    flamegpu::EnvironmentDirectedGraphDescription workplaceDigraphDesc= env.newDirectedGraph("WORKPLACE_DIGRAPH");
    // Graph vertices 
    workplaceDigraphDesc.newVertexProperty<float, 2>("bar");
    workplaceDigraphDesc.newEdgeProperty<int>("foo");


    // Workplace network variables. @todo - refactor to a separate network location?
    agent.newVariable<std::uint32_t>(person::v::WORKPLACE_IDX);
    agent.newVariable<std::uint32_t>(person::v::WORKPLACE_SIZE);

    // Random interaction network variables. @todo -refactor to separate location
    agent.newVariable<flamegpu::id_t, person::MAX_RANDOM_DAILY_INTERACTIONS>(person::v::RANDOM_INTERACTION_PARTNERS, {flamegpu::ID_NOT_SET});
    agent.newVariable<std::uint32_t>(person::v::RANDOM_INTERACTION_COUNT, 0u);
    agent.newVariable<std::uint32_t>(person::v::RANDOM_INTERACTION_COUNT_TARGET, 0u);

    // Add an environmental variable containing the sum of each agents target number of random interactions.
    // This is not the number of pair-wise interactions, but the number each individual would like.
    // there are cases where this value would be impossible to achieve for a given population.
    // I.e. if there are 2 agents, with targets of [2, 1]. This would be a target of 3, but only 1 interaction is possible
    env.newProperty<std::uint64_t>("RANDOM_INTERACTION_COUNT_SUM", 0u);

#if defined(FLAMEGPU_VISUALISATION)
    // @vis only
    agent.newVariable<float>(person::v::x);
    agent.newVariable<float>(person::v::y);
    agent.newVariable<float>(person::v::z);
#endif  // defined(FLAMEGPU_VISUALISATION)

    // Define relevant messages
    // Message list containing a persons current status for households (id, location, infection status)
    flamegpu::MessageBucket::Description householdStatusMessage = model.newMessage<flamegpu::MessageBucket>(person::message::household_status::_NAME);

    // Set the maximum bucket index to the population size. Ideally this would be household count, but that is not known at model definition time.
    // In the future this would be possible once https://github.com/FLAMEGPU/FLAMEGPU2/issues/710 is implemented
    householdStatusMessage.setUpperBound(params.n_total);

    // Add the agent id to the message.
    householdStatusMessage.newVariable<flamegpu::id_t>(person::message::household_status::ID);
    // Add the household index
    householdStatusMessage.newVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
    // Add a variable for the agent's infections status
    householdStatusMessage.newVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
    // Demographic?
    householdStatusMessage.newVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);

    // Message list containing a persons current status for workplaces (id, location, infection status)
    flamegpu::MessageBucket::Description workplaceStatusMessage = model.newMessage<flamegpu::MessageBucket>(person::message::workplace_status::_NAME);

    // Set the maximum bucket index to the population size, to the maximum number of workplace networks
    // @todo - this will be replaced with a per-person message when improved network messaging is implemented (where individuals will communicate with their direct network)
    workplaceStatusMessage.setUpperBound(exateppabm::population::WORKPLACE_COUNT);

    // Add the agent id to the message.
    workplaceStatusMessage.newVariable<flamegpu::id_t>(person::message::workplace_status::ID);
    // Add the household index
    workplaceStatusMessage.newVariable<std::uint32_t>(person::v::WORKPLACE_IDX);
    // Add a variable for the agent's infections status
    workplaceStatusMessage.newVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
    // Demographic?
    workplaceStatusMessage.newVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);

    // Message list containing a persons current status for random,daily interaction (id, location, infection status)
    // Uses an array message with 1 per agent
    flamegpu::MessageArray::Description randomNetworkStatusMessage = model.newMessage<flamegpu::MessageArray>(person::message::random_network_status::_NAME);

    // ID's are 1 indexed (0 is unset) so use + 1.
    // For ensembles this will need to be the largest n_total until https://github.com/FLAMEGPU/FLAMEGPU2/issues/710 is implemented
    randomNetworkStatusMessage.setLength(params.n_total + 1);

    // No need to add the agent ID to the message, it's implied by the bin count
    // randomNetworkStatusMessage.newVariable<flamegpu::id_t>(person::message::random_network_status::ID);

    // Add a variable for the agent's infections status
    randomNetworkStatusMessage.newVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
    // Agent's demographic
    randomNetworkStatusMessage.newVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);

    // Define agent functions
    // emit current status to the household
    flamegpu::AgentFunctionDescription emitHouseholdStatusDesc = agent.newFunction("emitHouseholdStatus", emitHouseholdStatus);
    emitHouseholdStatusDesc.setMessageOutput(person::message::household_status::_NAME);
    emitHouseholdStatusDesc.setInitialState(person::states::DEFAULT);
    emitHouseholdStatusDesc.setEndState(person::states::DEFAULT);

    // Interact with other agents in the household via their messages
    flamegpu::AgentFunctionDescription interactHouseholdDesc = agent.newFunction("interactHousehold", interactHousehold);
    interactHouseholdDesc.setMessageInput(person::message::household_status::_NAME);
    interactHouseholdDesc.setInitialState(person::states::DEFAULT);
    interactHouseholdDesc.setEndState(person::states::DEFAULT);

    // emit current status to the workplace
    flamegpu::AgentFunctionDescription emitWorkplaceStatusDesc = agent.newFunction("emitWorkplaceStatus", emitWorkplaceStatus);
    emitWorkplaceStatusDesc.setMessageOutput(person::message::workplace_status::_NAME);
    emitWorkplaceStatusDesc.setInitialState(person::states::DEFAULT);
    emitWorkplaceStatusDesc.setEndState(person::states::DEFAULT);

    // Interact with other agents in the workplace via their messages
    flamegpu::AgentFunctionDescription interactWorkplaceDesc = agent.newFunction("interactWorkplace", interactWorkplace);
    interactWorkplaceDesc.setMessageInput(person::message::workplace_status::_NAME);
    interactWorkplaceDesc.setInitialState(person::states::DEFAULT);
    interactWorkplaceDesc.setEndState(person::states::DEFAULT);

    // Update the daily random interactions
    // @todo - a GPU implementation will be needed for model scaling.
    flamegpu::HostFunctionDescription("updateRandomDailyNetworkIndices", updateRandomDailyNetworkIndices);

    // emit current status for random interactions
    flamegpu::AgentFunctionDescription emitRandomDailyNetworkStatusDesc = agent.newFunction("emitRandomDailyNetworkStatus", emitRandomDailyNetworkStatus);
    emitRandomDailyNetworkStatusDesc.setMessageOutput(person::message::random_network_status::_NAME);
    emitRandomDailyNetworkStatusDesc.setInitialState(person::states::DEFAULT);
    emitRandomDailyNetworkStatusDesc.setEndState(person::states::DEFAULT);

    // Interact with other agents in the random interactions
    flamegpu::AgentFunctionDescription interactRandomDailyNetworkDesc = agent.newFunction("interactRandomDailyNetwork", interactRandomDailyNetwork);
    interactRandomDailyNetworkDesc.setMessageInput(person::message::random_network_status::_NAME);
    interactRandomDailyNetworkDesc.setInitialState(person::states::DEFAULT);
    interactRandomDailyNetworkDesc.setEndState(person::states::DEFAULT);
}

void appendLayers(flamegpu::ModelDescription& model) {
    // Household interactions
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "emitHouseholdStatus");
    }
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "interactHousehold");
    }
    // Workplace interactions
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "emitWorkplaceStatus");
    }
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "interactWorkplace");
    }
    // Random interactions
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

}  // namespace person
}  // namespace exateppabm
