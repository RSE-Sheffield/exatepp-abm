#include "exateppabm/person.h"

#include <fmt/core.h>

#include "exateppabm/disease/SEIR.h"
#include "exateppabm/demographics.h"

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
 * @todo - prevent self-interactions
 * @todo - prevent duplicate interactions
 * @todo - more performance CPU implementation
 * @todo - GPU implementation of this (stable matching submodel?) 
 */

FLAMEGPU_HOST_FUNCTION(updateRandomDailyNetworkIndices) {
    // Get the current population of person agents

    // Get the sum of the per-agent random interaction count, so we can reserve a large enough vector

    // Build a vector of interactions, initialised to contain the id of each agent as many times as they require

    // Shuffle the vector of agent id's

    // Update agent data containing today's random interactions
    // @todo - support more than one interaction per agent per day.

}

/**
 * Agent function for person agents to emit their public information, i.e. infection status, for random daily network colleagues. This is put into a bucket key'd by the agent's ID.
 * 
 * @note the bucket message is from 1...N, rather than 0 to match ID index.
 */
FLAMEGPU_AGENT_FUNCTION(emitRandomDailyNetworkStatus, flamegpu::MessageNone, flamegpu::MessageBucket) {

    // output public properties to spatial message
    // Agent ID to avoid self messaging
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>(person::message::workplace_status::ID, FLAMEGPU->getID());

    FLAMEGPU->message_out.setVariable<disease::SEIR::InfectionStateUnderlyingType>(v::
    INFECTION_STATE, FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE));
    FLAMEGPU->message_out.setVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC, FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC));

    // Set the message key, the house hold idx for bucket messaging @Todo
    FLAMEGPU->message_out.setKey(FLAMEGPU->getID());
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
FLAMEGPU_AGENT_FUNCTION(interactRandomDailyNetwork, flamegpu::MessageBucket, flamegpu::MessageNone) {
    // Get my ID to avoid self messages
    const flamegpu::id_t id = FLAMEGPU->getID();

    // Get the probability of interaction within the workplace
    float p_daily_fraction_work = FLAMEGPU->environment.getProperty<float>("daily_fraction_work");

    // Get the probability of infection
    float p_s2e = FLAMEGPU->environment.getProperty<float>("p_interaction_susceptible_to_exposed");
     // Scale it for workplace interactions
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

        // @todo - support more than one random interaction, by looping over network edges or an agent array variable.
        flamegpu::id_t interactionPartnerID = FLAMEGPU->getVariable<flamegpu::id_t>(person::v::RANDOM_INTERACTION_PARTNER);

        // Iterate messages keyed by the InteractionPartner ID. There should only be one, so maybe use an Array Message @todo (but then it needs to be ID-1.)
        for (const auto &message : FLAMEGPU->message_in(interactionPartnerID)) {
            // Ignore self messages (can't infect oneself)
            if (message.getVariable<flamegpu::id_t>(message::random_network_status::ID) != id) {
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

    // Workplace network variables. @todo - refactor to a separate network location?
    agent.newVariable<std::uint32_t>(person::v::WORKPLACE_IDX);
    agent.newVariable<std::uint32_t>(person::v::WORKPLACE_SIZE);

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
    workplaceStatusMessage.setUpperBound(3);  // params.n_total);

    // Add the agent id to the message.
    workplaceStatusMessage.newVariable<flamegpu::id_t>(person::message::workplace_status::ID);
    // Add the household index
    workplaceStatusMessage.newVariable<std::uint32_t>(person::v::WORKPLACE_IDX);
    // Add a variable for the agent's infections status
    workplaceStatusMessage.newVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
    // Demographic?
    workplaceStatusMessage.newVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);

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
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "emitWorkplaceStatus");
    }
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "interactWorkplace");
    }
}

}  // namespace person
}  // namespace exateppabm
