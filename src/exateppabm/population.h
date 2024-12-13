#pragma once
#include <array>
#include <memory>
#include <random>
#include <vector>
#include "flamegpu/flamegpu.h"
#include "exateppabm/demographics.h"
#include "exateppabm/person.h"
#include "exateppabm/input.h"
#include "exateppabm/workplace.h"

namespace exateppabm {

namespace population {

/**
 * Define FLAME GPU init function to generate a population of agents for a simulation.
 * 
 * This method additionally stores several values in an anonymous namespace for use during the init function (which cannot take additional parameters)
 *
 * @param model the model the population is to be associated with (to get information about the person agent type from)
 * @param params the model parameters struct for this simulation
 * @param verbose if verbose output is enabled 
 */
void define(flamegpu::ModelDescription& model, const exateppabm::input::config params, const bool verbose);

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
