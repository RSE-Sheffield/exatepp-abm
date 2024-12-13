#include "exateppabm/workplace.h"

#include "flamegpu/flamegpu.h"
#include "exateppabm/demographics.h"
#include "exateppabm/disease.h"
#include "exateppabm/person.h"
#include "exateppabm/util.h"

namespace exateppabm {
namespace workplace {

/**
 * Namespace containing string constants related to the workplace message list
 */
namespace message_workplace_status {
constexpr char _NAME[] = "workplace_status";
}  // namespace message_workplace_status

/**
 * Agent function for person agents to emit their public information, i.e. infection status, to their workplace colleagues
 */
FLAMEGPU_AGENT_FUNCTION(emitWorkplaceStatus, flamegpu::MessageNone, flamegpu::MessageArray) {
    // Output public properties required for interactions to an array message, indexed by the agent's ID.

    FLAMEGPU->message_out.setVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::
    INFECTION_STATE, FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE));
    FLAMEGPU->message_out.setVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC, FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC));

    // Set the message key to be the agent ID for array access
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<flamegpu::id_t>(person::v::ID));
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

    // Get my workplace degree, if 0 there will be no interactions so do return.
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
    // auto workplaceIdx = FLAMEGPU->getVariable<WorkplaceUnderlyingType>(person::v::WORKPLACE_IDX);

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
                // No need to check workplace is correct, individuals only belong in one workplace
                // roll a dice to determine if this interaction should occur this day
                if (FLAMEGPU->random.uniform<float>() < p_daily_fraction_work) {
                    // Check if the other agent is infected
                    if (message.getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE) == disease::SEIR::InfectionState::Infected) {
                        // Roll a dice to determine if exposure occurred
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

    // Workplace index [0, WORKPLACE_COUNT)
    agent.newVariable<WorkplaceUnderlyingType>(person::v::WORKPLACE_IDX);
    // The number of workplace neighbours for this individual
    agent.newVariable<std::uint32_t>(person::v::WORKPLACE_OUT_DEGREE);

    // Message list containing a persons current status for workplaces (id, location, infection status)
    flamegpu::MessageArray::Description workplaceStatusMessage = model.newMessage<flamegpu::MessageArray>(message_workplace_status::_NAME);
    // Set the maximum message array index to the maximum expected ID (n_total + 1)
    workplaceStatusMessage.setLength(params.n_total + 1);
    // Add a variable for the agent's infections status
    workplaceStatusMessage.newVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
    // Demographic
    workplaceStatusMessage.newVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);

    // emit current status to the workplace message list
    flamegpu::AgentFunctionDescription emitWorkplaceStatusDesc = agent.newFunction("emitWorkplaceStatus", emitWorkplaceStatus);
    emitWorkplaceStatusDesc.setMessageOutput(message_workplace_status::_NAME);
    emitWorkplaceStatusDesc.setInitialState(person::states::DEFAULT);
    emitWorkplaceStatusDesc.setEndState(person::states::DEFAULT);

    // Interact with other agents in the workplace via their messages, based on the network graph
    flamegpu::AgentFunctionDescription interactWorkplaceDesc = agent.newFunction("interactWorkplace", interactWorkplace);
    interactWorkplaceDesc.setMessageInput(message_workplace_status::_NAME);
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



std::array<double, WORKPLACE_COUNT> getAdultWorkplaceCumulativeProbabilityArray(const double child_network_adults, const double elderly_network_adults, std::array<std::uint64_t, demographics::AGE_COUNT> n_per_age) {
    // Adults are assigned to a work place network randomly, using a probability distribution which (for a large enough sample) will match the target ratios of adults to other workplace members, based on the child_network_adults and elderly_network_adults parameters
    std::uint64_t n_0_9 = n_per_age[demographics::Age::AGE_0_9];
    std::uint64_t n_10_19 = n_per_age[demographics::Age::AGE_10_19];
    std::uint64_t n_70_79 = n_per_age[demographics::Age::AGE_70_79];
    std::uint64_t n_80_plus = n_per_age[demographics::Age::AGE_80];
    std::uint64_t n_adult = n_per_age[demographics::Age::AGE_20_29] + n_per_age[demographics::Age::AGE_30_39] + n_per_age[demographics::Age::AGE_40_49] + n_per_age[demographics::Age::AGE_50_59] + n_per_age[demographics::Age::AGE_60_69];

    // Initialise with the target number of adults in each network
    std::array<double, WORKPLACE_COUNT> p_adultPerWorkNetwork = {0};
    // p of each category is target number / number adults
    p_adultPerWorkNetwork[workplace::Workplace::WORKPLACE_SCHOOL_0_9] = (child_network_adults * n_0_9) / n_adult;
    p_adultPerWorkNetwork[workplace::Workplace::WORKPLACE_SCHOOL_10_19] = (child_network_adults * n_10_19) / n_adult;
    p_adultPerWorkNetwork[workplace::Workplace::WORKPLACE_70_79] = (elderly_network_adults * n_70_79) / n_adult;
    p_adultPerWorkNetwork[workplace::Workplace::WORKPLACE_80_PLUS] = (elderly_network_adults * n_80_plus) / n_adult;

    // p of being adult is then the remaining probability
    p_adultPerWorkNetwork[workplace::Workplace::WORKPLACE_ADULT] = 1.0 - std::accumulate(p_adultPerWorkNetwork.begin(), p_adultPerWorkNetwork.end(), 0.0);

    // Then convert to cumulative probability with an inclusive scan
    exateppabm::util::inclusive_scan(p_adultPerWorkNetwork.begin(), p_adultPerWorkNetwork.end(), p_adultPerWorkNetwork.begin());

    // make sure the top bracket ends in a value in case of floating point / rounding >= 1.0
    p_adultPerWorkNetwork[workplace::Workplace::WORKPLACE_80_PLUS] = 1.0;

    return p_adultPerWorkNetwork;
}

WorkplaceUnderlyingType generateWorkplaceForIndividual(const demographics::Age age, std::array<double, WORKPLACE_COUNT> p_adult_workplace, std::mt19937_64 & rng) {
    // Children, retired and elderly are assigned a network based on their age
    if (age == demographics::Age::AGE_0_9) {
        return workplace::Workplace::WORKPLACE_SCHOOL_0_9;
    } else if (age == demographics::Age::AGE_10_19) {
        return workplace::Workplace::WORKPLACE_SCHOOL_10_19;
    } else if (age == demographics::Age::AGE_70_79) {
        return workplace::Workplace::WORKPLACE_70_79;
    } else if (age == demographics::Age::AGE_80) {
        return workplace::Workplace::WORKPLACE_80_PLUS;
    } else {
        // Adults randomly sample using a cumulative probability distribution computed from population counts and model parameters
        std::uniform_real_distribution<double> work_network_dist(0.0, 1.0);
        float r = work_network_dist(rng);
        for (std::uint32_t i = 0; i < p_adult_workplace.size(); i++) {
            if (r < p_adult_workplace[i]) {
                return i;
            }
        }
        throw std::runtime_error("@todo - invalid cumulative probability distribution for p_adult_workplace?");
    }
}

}  // namespace workplace
}  // namespace exateppabm
