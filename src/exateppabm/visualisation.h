#pragma once

#include "flamegpu/flamegpu.h"

namespace exateppabm {
namespace visualisation {

void setup(bool enable, flamegpu::ModelDescription& model, flamegpu::CUDASimulation& simulation, bool paused, unsigned simulationSpeed);

void join();

}  // namespace visualisation

}  // namespace exateppabm
