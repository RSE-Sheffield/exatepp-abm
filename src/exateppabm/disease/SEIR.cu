#include "exateppabm/disease/SEIR.h"

#include "exateppabm/demographics.h"
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
    // Get the current timestep / day
    std::uint32_t today = FLAMEGPU->getStepCounter();

    // Get the current agents infection status
    auto infectionState = FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
    // Get the time they last changed state
    std::uint32_t dayOfLastStateChange = FLAMEGPU->getVariable<std::uint32_t>(person::v::INFECTION_STATE_CHANGE_DAY);
    // Get the duration of their state current state
    float stateDuration = FLAMEGPU->getVariable<float>(person::v::INFECTION_STATE_DURATION);
    // Ready to change state if today is past the next scheduled state change
    bool readyToChange = today >= dayOfLastStateChange + std::ceil(stateDuration);
    // For each different initial state, change if required and compute the next state's duration.
    if (infectionState == disease::SEIR::InfectionState::Exposed) {
        // Exposed to Infected, if enough time has passed
        if (readyToChange) {
            exposedToInfected(FLAMEGPU, infectionState);
        }
    } else if (infectionState == disease::SEIR::InfectionState::Infected) {
        // Infected to Recovered if enough time has passed
        if (readyToChange) {
            infectedToRecovered(FLAMEGPU, infectionState);
        }
    } else if (infectionState == disease::SEIR::InfectionState::Recovered) {
        // Recovered to Susceptible, if enough time has passed.
        if (readyToChange) {
            recoveredToSusceptible(FLAMEGPU, infectionState);
        }
    }
    return flamegpu::ALIVE;
}

void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params) {
    // Get a handle to the model environment description object
    flamegpu::EnvironmentDescription env = model.Environment();

    // Add a macro environment property (Atomically mutable) for tracking the cumulative number of infected individuals in each demographic.
    // @todo - should this be defined in time series instead? as its data collection not behaviour?
    env.newMacroProperty<std::uint32_t, demographics::AGE_COUNT>("total_infected_per_demographic");

    // Add a number of model parameters to the environment, initialised with the value from the configuration file
    // @todo - not all of these feel right here / add constexpr strings somewhere.
    env.newProperty<float>("mean_time_to_infected", params.mean_time_to_infected);
    env.newProperty<float>("sd_time_to_infected", params.sd_time_to_infected);
    env.newProperty<float>("mean_time_to_recovered", params.mean_time_to_recovered);
    env.newProperty<float>("sd_time_to_recovered", params.sd_time_to_recovered);
    env.newProperty<float>("mean_time_to_susceptible", params.mean_time_to_susceptible);
    env.newProperty<float>("sd_time_to_susceptible", params.sd_time_to_susceptible);

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
