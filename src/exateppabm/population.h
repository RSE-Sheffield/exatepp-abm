#pragma once
#include <memory>
#include "flamegpu/flamegpu.h"
#include "exateppabm/demographics.h"
#include "exateppabm/person.h"
#include "exateppabm/input.h"

namespace exateppabm {

namespace population {

/**
 * Generate a population of individual person agents for a given simulation configuration
 *
 * @param model the model the population is to be associated with (to get information about the person agent type from)
 * @param config the model parameters struct for this simulation
 * @param verbose if verbose output is enabled 
 * @param env_width the 2D environment width used for temporary behaviour / initialisation. To be removed once networks added.
 * @param interactionRadius Temporary interaction radius for this simulation. To be removed
 */
std::unique_ptr<flamegpu::AgentVector> generate(flamegpu::ModelDescription& model, const exateppabm::input::config config, const bool verbose, const float env_width, const float interactionRadius);

/**
 * Get the number of agents per demographic which were initialised to be infected, for the most recent call to generate.
 * 
 * This is a workaround to make these values available in a FLAMEGPU_INIT_FUNC.
 * 
 * @todo - refactor this during generation of a realistic population of agents
 * @todo - refactor to be able to do this in a thread safe way from within an INIT function.
 * @note - 9 element std array, don't mind this creating a copy for a one time use method.
 * @return std::array containing the number of each demographic which were initialised to be infected
 */
std::array<std::uint64_t, exateppabm::demographics::AGE_COUNT> getPerDemographicInitialInfectionCount();


}  // namespace population

}  // namespace exateppabm
