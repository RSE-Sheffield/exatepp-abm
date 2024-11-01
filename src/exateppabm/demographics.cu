#include "exateppabm/demographics.h"

#include <cstdint>

#include "flamegpu/flamegpu.h"
#include "exateppabm/input.h"

namespace exateppabm {
namespace demographics {

void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params) {
    // Get a handle to the model environment description object
    flamegpu::EnvironmentDescription env = model.Environment();

    // Define a new environment property array variable, with the per-demographic infection susceptibility modifier
    // @todo - store this environment property variable name somewhere.
    env.newProperty<float, demographics::AGE_COUNT>("relative_susceptibility_per_demographic", {
        params.relative_susceptibility_0_9,
        params.relative_susceptibility_10_19,
        params.relative_susceptibility_20_29,
        params.relative_susceptibility_30_39,
        params.relative_susceptibility_40_49,
        params.relative_susceptibility_50_59,
        params.relative_susceptibility_60_69,
        params.relative_susceptibility_70_79,
        params.relative_susceptibility_80
    }, true);
}

}  // namespace demographics
}  // namespace exateppabm
