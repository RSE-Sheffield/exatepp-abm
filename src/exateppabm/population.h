#pragma once
#include <memory>
#include "flamegpu/flamegpu.h"
#include "exateppabm/person.h"
#include "exateppabm/input.h"

namespace exateppabm {

namespace population {

/**
 * Generate a population of individual person agents for a given simulation configuration
 *
 * @param model the model the population is to be associated with (to get information about the person agent type from)
 * @param env_width the 2D environment width used for temporary behaviour / initialisation. To be removed once networks added.
 * @param interactionRadius Temporary interaction radius for this simulation. To be removed
 */
std::unique_ptr<flamegpu::AgentVector> generate(flamegpu::ModelDescription& model, const exateppabm::input::config config, const float env_width, const float interactionRadius);

}  // namespace population

}  // namespace exateppabm
