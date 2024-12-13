#include "exateppabm/random_interactions.h"

#include "flamegpu/flamegpu.h"
#include "exateppabm/demographics.h"
#include "exateppabm/disease.h"
#include "exateppabm/person.h"

namespace exateppabm {
namespace workplace {

/**
 * Agent function for person agents to emit their public information, i.e. infection status, to their workplace colleagues
 */
FLAMEGPU_AGENT_FUNCTION(emitWorkplaceStatus, flamegpu::MessageNone, flamegpu::MessageArray) {
    // output public properties to bucket message, keyed by workplace
    // Agent ID to avoid self messaging
    flamegpu::id_t id = FLAMEGPU->getVariable<flamegpu::id_t>(person::v::ID);
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>(person::message::workplace_status::ID, id);

    // workplace index
    // @todo - typedef or using statement for the workplace index type?
    std::uint32_t workplaceIdx = FLAMEGPU->getVariable<std::uint32_t>(person::v::WORKPLACE_IDX);
    FLAMEGPU->message_out.setVariable<std::uint32_t>(person::v::WORKPLACE_IDX, workplaceIdx);

    FLAMEGPU->message_out.setVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::
    INFECTION_STATE, FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE));
    FLAMEGPU->message_out.setVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC, FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC));

    // Set the message key, the house hold idx for bucket messaging @Todo
    FLAMEGPU->message_out.setIndex(id);
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
FLAMEGPU_AGENT_FUNCTION(interactWorkplace, flamegpu::MessageArray, flamegpu::MessageNone) {
    // Get my ID
    const flamegpu::id_t id = FLAMEGPU->getVariable<flamegpu::id_t>(person::v::ID);

    // Get my workplace degree, if 0 return.
    const std::uint32_t degree = FLAMEGPU->getVariable<std::uint32_t>(person::v::WORKPLACE_OUT_DEGREE);
    if (degree == 0) {
        return flamegpu::ALIVE;
    }

    // Get the probability of interaction within the workplace
    float p_daily_fraction_work = FLAMEGPU->environment.getProperty<float>("daily_fraction_work");

    // Get the probability of an interaction leading to exposure
    float p_s2e = FLAMEGPU->environment.getProperty<float>("p_interaction_susceptible_to_exposed");
     // Scale it for workplace interactions
    p_s2e *= FLAMEGPU->environment.getProperty<float>("relative_transmission_occupation");

    // Get my workplace/network index
    auto workplaceIdx = FLAMEGPU->getVariable<std::uint32_t>(person::v::WORKPLACE_IDX);

    // Get my age demographic
    auto demographic = FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);
    // Get my susceptibility modifier and modify it.
    float relativeSusceptibility = FLAMEGPU->environment.getProperty<float, demographics::AGE_COUNT>("relative_susceptibility_per_demographic", demographic);
    // Scale the probability of transmission for my age demographic
    p_s2e *= relativeSusceptibility;

    // Check if the current individual is susceptible to being infected
    auto infectionState = FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);

    // @todo - this will need to change for contact tracing, the message interaction will need to occur regardless.
    if (infectionState == disease::SEIR::Susceptible) {
        // Variable to store the duration of the exposed phase (if exposed)
        float stateDuration = 0.f;
        // Iterate my downstream neighbours (the graph is undirected, so no need to iterate in and out
        auto workplaceGraph = FLAMEGPU->environment.getDirectedGraph("WORKPLACE_DIGRAPH");
        std::uint32_t myVertexIndex = workplaceGraph.getVertexIndex(id);
        for (auto& edge : workplaceGraph.outEdges(myVertexIndex)) {
            // Get the neighbour vertex index and then neighbours agent Id from the edge
            std::uint32_t otherIndex = edge.getEdgeDestination();
            flamegpu::id_t otherID = workplaceGraph.getVertexID(otherIndex);
            // if the ID is not self and not unset
            if (otherID != id && otherID != flamegpu::ID_NOT_SET) {
                // Get the message handle
                const auto &message = FLAMEGPU->message_in.at(otherID);
                // Ignore messages from other workplaces
                if (message.getVariable<std::uint32_t>(person::v::WORKPLACE_IDX) == workplaceIdx) {
                    // roll a dice to determine if this interaction should occur this day
                    if (FLAMEGPU->random.uniform<float>() < p_daily_fraction_work) {
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

    // Environment for the transmission scale factor within workplaces
    env.newProperty<float>("relative_transmission_occupation", params.relative_transmission_occupation);

    // env var for the fraction of people in the same work network to interact with
    env.newProperty<float>("daily_fraction_work", params.daily_fraction_work);

    // Workplace environment directed graph
    // This single graph contains workplace information for all individuals, and is essentially 5 unconnected sub graphs.
    // There are no explicit vertex or edge properties, just the structure is required
    flamegpu::EnvironmentDirectedGraphDescription workplaceDigraphDesc = env.newDirectedGraph("WORKPLACE_DIGRAPH");

    // Get a handle to the existing person agent type, which should have already been defined.
    flamegpu::AgentDescription agent = model.Agent(person::NAME);

    // Workplace network variables. @todo - refactor to a separate network location?
    agent.newVariable<std::uint32_t>(person::v::WORKPLACE_IDX);
    agent.newVariable<std::uint32_t>(person::v::WORKPLACE_OUT_DEGREE);


    // Message list containing a persons current status for workplaces (id, location, infection status)
    flamegpu::MessageArray::Description workplaceStatusMessage = model.newMessage<flamegpu::MessageArray>(person::message::workplace_status::_NAME);
    // Set the maximum message array index to the maximum expected ID (n_total + 1)
    workplaceStatusMessage.setLength(params.n_total + 1);
    // Add the agent id to the message.
    workplaceStatusMessage.newVariable<flamegpu::id_t>(person::message::workplace_status::ID);
    // Add the household index
    workplaceStatusMessage.newVariable<std::uint32_t>(person::v::WORKPLACE_IDX);
    // Add a variable for the agent's infections status
    workplaceStatusMessage.newVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
    // Demographic?
    workplaceStatusMessage.newVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);

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
        layer.addAgentFunction(person::NAME, "emitWorkplaceStatus");
    }
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "interactWorkplace");
    }
}

}  // namespace workplace
}  // namespace exateppabm
