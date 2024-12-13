#include "exateppabm/person.h"

#include <fmt/core.h>

#include <vector>

#include "exateppabm/disease/SEIR.h"
#include "exateppabm/demographics.h"
#include "exateppabm/household.h"
#include "exateppabm/workplace.h"
#include "exateppabm/random_interactions.h"
#include "exateppabm/visualisation.h"

namespace exateppabm {
namespace person {

void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params) {
    // Define related model environment properties (@todo - abstract these somewhere more appropriate at a later date)
    flamegpu::EnvironmentDescription env = model.Environment();

    // @todo this should probably be refactored elsewhere (although used in this file currently)
    // Define the base probability of being exposed if interacting with an infected individual, used in multiple locations
    env.newProperty<float>("p_interaction_susceptible_to_exposed", params.p_interaction_susceptible_to_exposed);

    // Define the agent type
    flamegpu::AgentDescription agent = model.newAgent(person::NAME);

    // Define states
    agent.newState(person::states::DEFAULT);

    // Define variables
    // Custom ID, as getID cannot be relied upon to start at 1.
    agent.newVariable<flamegpu::id_t>(person::v::ID, flamegpu::ID_NOT_SET);

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

    // Call define methods for specific behaviours related to person, which will add additional agent variables, messages and functions to the model

    // Define household related agent variables, functions, etc
    household::define(model, params);

    // Define workplace related agent variables, functions, etc
    workplace::define(model, params);

    // Define daily random interaction related agent variables, functions, etc
    random_interactions::define(model, params);

    // Define visualisation specific behaviours
    visualisation::define(model, params);
}

void appendLayers(flamegpu::ModelDescription& model) {
    // Household interactions
    household::appendLayers(model);

    // Workplace interactions
    workplace::appendLayers(model);

    // Random interactions
    random_interactions::appendLayers(model);
}

}  // namespace person
}  // namespace exateppabm
