#pragma once

#include "flamegpu/flamegpu.h"

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

}  // namespace visualisation
}  // namespace exateppabm
