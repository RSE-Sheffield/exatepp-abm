#include "exateppabm/person.h"

#include <fmt/core.h>

#include "exateppabm/disease/SEIR.h"
#include "exateppabm/demographics.h"

namespace exateppabm {
namespace person {

/**
 * Agent function for person agents to emit their public information, i.e. infection status
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

        // Location if used for comms
        // FLAMEGPU->message_out.setVariable<float>(v::x, FLAMEGPU->getVariable<float>(v::x));
        // FLAMEGPU->message_out.setVariable<float>(v::y, FLAMEGPU->getVariable<float>(v::y));
        // FLAMEGPU->message_out.setVariable<float>(v::z, FLAMEGPU->getVariable<float>(v::z));

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

    if (infectionState == disease::SEIR::Susceptible) {
        // Agent position
        float agent_x = FLAMEGPU->getVariable<float>(v::x);
        float agent_y = FLAMEGPU->getVariable<float>(v::y);
        // float agent_z = FLAMEGPU->getVariable<float>(v::z);

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

void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params, const float width, const float interactionRadius) {
    // Define related model environment properties (@todo - abstract these somewhere more appropriate at a later date)
    flamegpu::EnvironmentDescription env = model.Environment();
    env.newProperty<float>("INFECTION_INTERACTION_RADIUS", interactionRadius);
    // Define an infection probabiltiy. @todo this should be from the config file.
    env.newProperty<float>("p_interaction_susceptible_to_exposed", params.p_interaction_susceptible_to_exposed);

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

    // age demographic
    // @todo make this an enum, and update uses of it, but flame's templating disagrees?
    agent.newVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC, exateppabm::demographics::Age::AGE_0_9);

    // Household network variables. @todo - refactor to a separate network location?
    agent.newVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
    agent.newVariable<std::uint8_t>(person::v::HOUSEHOLD_SIZE);

    // @todo - temp or vis only?
    agent.newVariable<float>(person::v::x);
    agent.newVariable<float>(person::v::y);
    agent.newVariable<float>(person::v::z);

    // Define relevant messages
    // Message list containing a persons current status (id, location, infection status)
    fmt::print("@todo - use bucket messaging not brute force for household comms.\n");  // However, this requires knowing the maximum number of houses here, which is not known yet or for ensembles. Worst case would be the number of people?
    flamegpu::MessageBruteForce::Description statusMessage = model.newMessage<flamegpu::MessageBruteForce>(person::message::household_status::_NAME);

    // Add the agent id to the message.
    statusMessage.newVariable<flamegpu::id_t>(person::message::household_status::ID);
    // Add the household index
    statusMessage.newVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
    // Add a variable for the agent's infections status
    statusMessage.newVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
    // Demographic?
    statusMessage.newVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);

    // Define agent functions
    // emit current status
    flamegpu::AgentFunctionDescription emitHouseholdStatusDesc = agent.newFunction("emitStatus", emitHouseholdStatus);
    emitHouseholdStatusDesc.setMessageOutput(person::message::household_status::_NAME);
    emitHouseholdStatusDesc.setInitialState(person::states::DEFAULT);
    emitHouseholdStatusDesc.setEndState(person::states::DEFAULT);

    // Interact with other agents via their messages
    flamegpu::AgentFunctionDescription interactHouseholdDesc = agent.newFunction("interact", interactHousehold);
    interactHouseholdDesc.setMessageInput(person::message::household_status::_NAME);
    interactHouseholdDesc.setInitialState(person::states::DEFAULT);
    interactHouseholdDesc.setEndState(person::states::DEFAULT);
}

void appendLayers(flamegpu::ModelDescription& model) {
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "emitStatus");
    }
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "interact");
    }
}

}  // namespace person
}  // namespace exateppabm
