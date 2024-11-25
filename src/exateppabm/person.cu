#include "exateppabm/person.h"

#include <fmt/core.h>

#include "exateppabm/disease/SEIR.h"
#include "exateppabm/demographics.h"

namespace exateppabm {
namespace person {

/**
 * Agent function for person agents to emit their public information, i.e. infection status to their household
 */
FLAMEGPU_AGENT_FUNCTION(emitHouseholdStatus, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    // Households of 1 don't need to do any messaging, there is no one to infect
    std::uint8_t householdSize = FLAMEGPU->getVariable<std::uint8_t>(v::HOUSEHOLD_SIZE);
    if (householdSize > 1) {
        // output public properties to spatial message
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
        // FLAMEGPU->message_out.setKey(householdIdx);
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
FLAMEGPU_AGENT_FUNCTION(interactHousehold, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
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

        // Iterate messages from anyone within my spatial neighbourhood (i.e. cuboid not sphere)
        for (const auto &message : FLAMEGPU->message_in) {
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
FLAMEGPU_AGENT_FUNCTION(emitWorkplaceStatus, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    // workplaces of 1 don't need to do any messaging, there is no one to infect
    std::uint32_t workplaceSize = FLAMEGPU->getVariable<std::uint32_t>(v::WORKPLACE_SIZE);
    if (workplaceSize > 1) {
        // output public properties to spatial message
        // Agent ID to avoid self messaging
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>(person::message::workplace_status::ID, FLAMEGPU->getID());

        // Household index
        // @todo - typedef or using statement for the household index type?
        std::uint32_t workplaceIdx = FLAMEGPU->getVariable<std::uint32_t>(v::WORKPLACE_IDX);
        FLAMEGPU->message_out.setVariable<std::uint32_t>(v::WORKPLACE_IDX, workplaceIdx);

        FLAMEGPU->message_out.setVariable<disease::SEIR::InfectionStateUnderlyingType>(v::
        INFECTION_STATE, FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE));
        FLAMEGPU->message_out.setVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC, FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC));

        // Set the message key, the house hold idx for bucket messaging @Todo
        // FLAMEGPU->message_out.setKey(householdIdx);
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
FLAMEGPU_AGENT_FUNCTION(interactWorkplace, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
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

        // Iterate messages from anyone within my spatial neighbourhood (i.e. cuboid not sphere)
        for (const auto &message : FLAMEGPU->message_in) {
            // Ignore self messages (can't infect oneself)
            if (message.getVariable<flamegpu::id_t>(message::household_status::ID) != id) {
                // Ignore messages from other households
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

void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params) {
    // Define related model environment properties (@todo - abstract these somewhere more appropriate at a later date)
    flamegpu::EnvironmentDescription env = model.Environment();

    // @todo this should probably be refactored elsewhere (although used in this file currently)
    // Define the base probability of being exposed if interacting with an infected individual
    env.newProperty<float>("p_interaction_susceptible_to_exposed", params.p_interaction_susceptible_to_exposed);
    // define the per interaction type scale factors as env var
    env.newProperty<float>("relative_transmission_household", params.relative_transmission_household);
    env.newProperty<float>("relative_transmission_occupation", params.relative_transmission_occupation);
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
    fmt::print("@todo - use bucket messaging not brute force for household comms.\n");  // However, this requires knowing the maximum number of houses here, which is not known yet or for ensembles. Worst case would be the number of people?
    flamegpu::MessageBruteForce::Description householdStatusMessage = model.newMessage<flamegpu::MessageBruteForce>(person::message::household_status::_NAME);

    // Add the agent id to the message.
    householdStatusMessage.newVariable<flamegpu::id_t>(person::message::household_status::ID);
    // Add the household index
    householdStatusMessage.newVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
    // Add a variable for the agent's infections status
    householdStatusMessage.newVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
    // Demographic?
    householdStatusMessage.newVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);

    // Message list containing a persons current status for workplaces (id, location, infection status)
    fmt::print("@todo - use bucket messaging not brute force for household comms.\n");  // However, this requires knowing the maximum number of houses here, which is not known yet or for ensembles. Worst case would be the number of people?
    flamegpu::MessageBruteForce::Description workplaceStatusMessage = model.newMessage<flamegpu::MessageBruteForce>(person::message::workplace_status::_NAME);

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
