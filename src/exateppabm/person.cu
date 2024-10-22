#include "exateppabm/person.h"

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
    FLAMEGPU->message_out.setVariable<std::uint32_t>(v::
    INFECTED, FLAMEGPU->getVariable<std::uint32_t>(v::INFECTED));
    FLAMEGPU->message_out.setVariable<std::uint8_t>(v::DEMOGRAPHIC, FLAMEGPU->getVariable<std::uint8_t>(v::DEMOGRAPHIC));

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
    float inf_p = FLAMEGPU->environment.getProperty<float>("INFECTION_PROBABILITY");

    // Get my ID to avoid self messages
    const flamegpu::id_t id = FLAMEGPU->getID();

    // Check if I am already infected. Interactions are one way for now.
    std::uint32_t infected = FLAMEGPU->getVariable<std::uint32_t>(v::INFECTED);

    if(!infected){
        // Agent position
        float agent_x = FLAMEGPU->getVariable<float>(v::x);
        float agent_y = FLAMEGPU->getVariable<float>(v::y);
        // float agent_z = FLAMEGPU->getVariable<float>(v::z);

        // Iterate messages from anyone within my spatial neighbourhood (i.e. cuboid not sphere)
        for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y)) {
            // Ignore self messages (can't infect oneself)
            if (message.getVariable<flamegpu::id_t>(message::v::STATUS_ID) != id) {
                // Check if the other agent is infected
                if(message.getVariable<std::uint32_t>(v::INFECTED)){
                    // Roll a dice
                    float r = FLAMEGPU->random.uniform<float>();
                    if (r < inf_p) {
                        // I have become infected.
                        infected = true;
                        break;
                    }
                }
            }
        }
        // Update my status in my global device memory (outside the loop.)
        if(infected) {
            FLAMEGPU->setVariable<std::uint32_t>(v::INFECTED, infected);
        }
    }


    
    return flamegpu::ALIVE;
}


void define(flamegpu::ModelDescription& model, const float width, const float interactionRadius) {
    // Define related model environment properties (@todo - abstract these somewhere more appropriate at a later date)
    flamegpu::EnvironmentDescription env = model.Environment();
    env.newProperty<float>("INFECTION_INTERACTION_RADIUS", interactionRadius);
    // Define an infection probabiltiy. @todo this should be from the config file.
    env.newProperty<float>("INFECTION_PROBABILITY", 0.01f);


    // Define the agent type
    flamegpu::AgentDescription agent = model.newAgent(exateppabm::person::NAME);

    // Define states
    agent.newState(exateppabm::person::states::DEFAULT);

    // Define variables
    // @todo - make this an enum / store it elsewhere. 
    // This has to be 32 bit for vis purposes :(. 
    agent.newVariable<std::uint32_t>(exateppabm::person::v::INFECTED, 0);
    // @todo make this an enum, and update uses of it, but flame's templating disagrees?
    agent.newVariable<std::uint8_t>(exateppabm::person::v::DEMOGRAPHIC, static_cast<uint8_t>(exateppabm::person::Demographic::AGE_0_9));

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
    statusMessage.newVariable<std::uint32_t>(exateppabm::person::v::INFECTED);
    // Demographic? 
    statusMessage.newVariable<std::uint8_t>(exateppabm::person::v::DEMOGRAPHIC);


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
