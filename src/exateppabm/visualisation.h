#pragma once

#include <memory>

#include "flamegpu/flamegpu.h"
#include "exateppabm/input.h"

namespace exateppabm {
namespace visualisation {

/**
 * Setup a FLAME GPU 2 visualisation for a specific simulation instance
 * 
 * @param enable if vis should be enabled
 * @param model the model the simulation is based on
 * @param simulation FLAME GPU 2 simulation instance to be visualised
 * @param paused if the simulation should start paused or not
 * @param simulationSpeed target simulation rate in steps per second, to ensure changes are visible.
 */
void setup(bool enable, flamegpu::ModelDescription& model, flamegpu::CUDASimulation& simulation, bool paused, unsigned simulationSpeed);

/**
 * Join an flame gpu 2 visualisation thread (if active). I.e. ensure the visualisation window has been closed before continuing.
 */
void join();

/**
 * Set visualisation specific agent variables for a given population
 * 
 * I.e. set the x/y/z based on the their householdIdx
 * 
 * This is more expensive as a separate loop, but only a one time cost and visualsiation runs are not as sensitive to performance
 * 
 * @param model flame gpu model description object
 * @param config simulation configuration parameters
 * @param pop FLAME GPU person population object, with partially initialised population data
 * @param householdCount the number of generated households
 */
void initialiseAgentPopulation(const flamegpu::ModelDescription& model, const exateppabm::input::config config, std::unique_ptr<flamegpu::AgentVector> & pop, const std::uint32_t householdCount);

}  // namespace visualisation
}  // namespace exateppabm
