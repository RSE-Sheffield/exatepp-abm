#pragma once
#include <memory>
#include "flamegpu/flamegpu.h"
#include "person.h"
#include "input.h"

namespace exateppabm {

namespace population {

std::unique_ptr<flamegpu::AgentVector> generate(flamegpu::ModelDescription& model, const exateppabm::input::config config, const float env_width, const float interactionRadius);

}  // namespsace population

} // namespace exateppabm