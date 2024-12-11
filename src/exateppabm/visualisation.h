#pragma once

#include <memory>
#include <tuple>

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
 * Get visualistion properties for an agent, given model parameters and thier household index
 * 
 * This has had to be refactored to workaround a flame gpu limitation
 * 
 * @param householdCount the number of generated households
 * @param householdIdx, the index of the current household
 * @param idxWithinHousehold the index within the household for the agent.
 * @return tuple of agent properties
 */
std::tuple<float, float, float> getAgentXYZ(const std::uint32_t householdCount, const std::uint32_t householdIdx, const std::uint8_t idxWithinHousehold);

}  // namespace visualisation
}  // namespace exateppabm
