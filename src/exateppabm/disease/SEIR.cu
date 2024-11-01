#include "exateppabm/disease/SEIR.h"

#include "exateppabm/person.h"

namespace exateppabm {
namespace disease {
namespace SEIR {

/**
 * FLAME GPU Agent function which progresses the disease of the current agent.
 *
 * @todo -include agent demographics in this?
 * @todo - env based parameters?
 */
FLAMEGPU_AGENT_FUNCTION(progressDisease, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Get the current agents infection status
    auto infectionState = FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);

    // Get the agent's demographic
    auto demographic_idx = FLAMEGPU->getVariable<std::uint8_t>(person::v::DEMOGRAPHIC);

    // Get a handle to the total_infected_per_demographic macro env property
    auto totalInfectedPerDemographic = FLAMEGPU->environment.getMacroProperty<std::uint32_t, person::DEMOGRAPHIC_COUNT>("total_infected_per_demographic");

    if (infectionState == disease::SEIR::InfectionState::Exposed) {
        float r = FLAMEGPU->random.uniform<float>();
        // Chance to become infected
        if (r < 0.10) {
            infectionState = disease::SEIR::InfectionState::Infected;
            // Atomically update the number of infected individuals  for the current individuals demographics when they transition into the infection state
            totalInfectedPerDemographic[demographic_idx]++;
        // Chance to become susceptible
        } else if (r < 0.50) {
            infectionState = disease::SEIR::InfectionState::Susceptible;
        // otherwise remain exposed
        } else {
            // @todo - don't remain exposed forever?
            // noop
        }
    } else if (infectionState == disease::SEIR::InfectionState::Infected) {
        float r = FLAMEGPU->random.uniform<float>();
        // Chance to become Recovered
        if (r < 0.01) {
            infectionState = disease::SEIR::InfectionState::Recovered;
        }
    }

    // @todo use other states

    // Update global agent variables from local (in register) values.
    FLAMEGPU->setVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE, infectionState);

    return flamegpu::ALIVE;
}

void define(flamegpu::ModelDescription& model) {
    // Get a handle to the model environment description object
    flamegpu::EnvironmentDescription env = model.Environment();

    // Add a macro environment property (Atomically mutable) for tracking the cumulative number of infected individuals in each demographic.
    // @todo - should this be defined in time series instead? as its data collection not behaviour?
    env.newMacroProperty<std::uint32_t, person::DEMOGRAPHIC_COUNT>("total_infected_per_demographic");



    // Define the temporary, hardcoded spatial infection interaction radius?
    // env.setProperty<float>("INFECTION_INTERACTION_RADIUS", interactionRadius);

    // Get a handle for the Person agent type
    flamegpu::AgentDescription person = model.Agent(exateppabm::person::NAME);

    // @todo - add agent variables here? Need to decide where is best, depending on how things are used / accessed?

    // Define disease-only agent functions and message lists
    // @todo - consider using FLAME GPU 2 states along with the DiseaseState to improve performance if possible?
    // @todo - add function name to a namespace
    flamegpu::AgentFunctionDescription diseaseProgression = person.newFunction("progressDisease", progressDisease);
    diseaseProgression.setInitialState(exateppabm::person::states::DEFAULT);
    diseaseProgression.setEndState(exateppabm::person::states::DEFAULT);
}

void appendLayers(flamegpu::ModelDescription& model) {
    // Add the disease progression function for person agents
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(exateppabm::person::NAME, "progressDisease");
    }
}

}  // namespace SEIR
}  // namespace disease
}  // namespace exateppabm
