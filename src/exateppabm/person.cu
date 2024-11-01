#include "exateppabm/person.h"

#include "exateppabm/disease/SEIR.h"
#include "exateppabm/demographics.h"

namespace exateppabm {
namespace person {

/**
 * Agent function for person agents to emit their public information, i.e. infection status
 */
FLAMEGPU_AGENT_FUNCTION(emitStatus, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    // output public properties to spatial message
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>(person::message::v::STATUS_ID, FLAMEGPU->getID());
    // Location which is also used for comms
    FLAMEGPU->message_out.setVariable<float>(v::x, FLAMEGPU->getVariable<float>(v::x));
    FLAMEGPU->message_out.setVariable<float>(v::y, FLAMEGPU->getVariable<float>(v::y));
    // FLAMEGPU->message_out.setVariable<float>(v::z, FLAMEGPU->getVariable<float>(v::z));

    // And their
    FLAMEGPU->message_out.setVariable<disease::SEIR::InfectionStateUnderlyingType>(v::
    INFECTION_STATE, FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(v::INFECTION_STATE));
    FLAMEGPU->message_out.setVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC, FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(v::AGE_DEMOGRAPHIC));

    return flamegpu::ALIVE;
}

/**
 * Very naive agent interaction for infection spread via "contact"
 *
 * Agents iterate messages from their local neighbours.
 * If their neighbour was infected, RNG sample to determine if "i" become infected.
 */
FLAMEGPU_AGENT_FUNCTION(interact, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    // Get the probability of infection
    float p_s2e = FLAMEGPU->environment.getProperty<float>("p_interaction_susceptible_to_exposed");

    // Get my ID to avoid self messages
    const flamegpu::id_t id = FLAMEGPU->getID();

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
        for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y)) {
            // Ignore self messages (can't infect oneself)
            if (message.getVariable<flamegpu::id_t>(message::v::STATUS_ID) != id) {
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
    flamegpu::AgentDescription agent = model.newAgent(exateppabm::person::NAME);

    // Define states
    agent.newState(exateppabm::person::states::DEFAULT);

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
    agent.newVariable<demographics::AgeUnderlyingType>(exateppabm::person::v::AGE_DEMOGRAPHIC, exateppabm::demographics::Age::AGE_0_9);

    // @todo - temp or vis only?
    agent.newVariable<float>(exateppabm::person::v::x);
    agent.newVariable<float>(exateppabm::person::v::y);
    agent.newVariable<float>(exateppabm::person::v::z);

    // Define relevant messages
    // Message list containing a persons current status (id, location, infection status)
    flamegpu::MessageSpatial2D::Description statusMessage = model.newMessage<flamegpu::MessageSpatial2D>(exateppabm::person::message::STATUS);
    // Set the range and bounds for the spatial component of the status message.
    // This effectively limits the range of the simulation. For now we pass these values in.
    statusMessage.setRadius(env.getProperty<float>("INFECTION_INTERACTION_RADIUS"));
    statusMessage.setMin(0, 0);
    statusMessage.setMax(width, width);

    // Add the id to the message. x, y, z are implicit
    statusMessage.newVariable<flamegpu::id_t>(person::message::v::STATUS_ID);
    // Add a variable for the agent's infections status
    statusMessage.newVariable<disease::SEIR::InfectionStateUnderlyingType>(exateppabm::person::v::INFECTION_STATE);
    // Demographic?
    statusMessage.newVariable<demographics::AgeUnderlyingType>(exateppabm::person::v::AGE_DEMOGRAPHIC);

    // Define agent functions
    // emit current status
    flamegpu::AgentFunctionDescription emitStatusDesc = agent.newFunction("emitStatus", emitStatus);
    emitStatusDesc.setMessageOutput(exateppabm::person::message::STATUS);
    emitStatusDesc.setInitialState(exateppabm::person::states::DEFAULT);
    emitStatusDesc.setEndState(exateppabm::person::states::DEFAULT);

    // Interact with other agents via their messages
    flamegpu::AgentFunctionDescription interactDesc = agent.newFunction("interact", interact);
    interactDesc.setMessageInput(exateppabm::person::message::STATUS);
    emitStatusDesc.setInitialState(exateppabm::person::states::DEFAULT);
    emitStatusDesc.setEndState(exateppabm::person::states::DEFAULT);
}

void appendLayers(flamegpu::ModelDescription& model) {
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(exateppabm::person::NAME, "emitStatus");
    }
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(exateppabm::person::NAME, "interact");
    }
}

}  // namespace person
}  // namespace exateppabm
