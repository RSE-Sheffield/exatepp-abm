#pragma once

#include <cstdint>
#include "flamegpu/flamegpu.h"
#include "exateppabm/input.h"

namespace exateppabm {
namespace disease {

/**
 * Methods and variables related to SEIR modelling, with no birth or death included
 * 
 * This is intended to demonstrate the type of model which can be implemented and how, with more complex disease modelling following a similar pattern.
 */
namespace SEIR {

/**
 * Type definition for the underlying type used for the InfectionState Enum.
 * @note - this is a uint32_t for now so it can be a FLAME GPU 2 visualisation colour index. Ideally it would jus be an uint_8t
 */

typedef std::uint32_t InfectionStateUnderlyingType;

/**
 * Enum representing the different states of the model. 
 * Unfortunately unable to use a enum class for strong typing without excessive casting being required when passing to/from templated FLAMEGPU 2 methods.
 * 
 * @note - the static_casting required might be worthwhile if multiple infection models are implemented, as non-class enums are in the global scope without any prefixing.
 * @note - Possibly just define __host__ __device__ methods to/from_uint8() rather than having the cast? Not sure <type_traits> is available on device for a host device to_underlying pre c++23
 * @note - not just named "state" to reduce confusion with FLAME GPU 2 agent states.
 */
enum InfectionState : InfectionStateUnderlyingType {
    Susceptible = 0,
    Exposed = 1,
    Infected = 2,
    Recovered = 3
};


/**
 * Add required parts to a FLAME GPU 2 model for this implementation.
 * 
 * @note - this is likely to be refactored several times
 * @param model the FLAME GPU Model description to mutate
 * @param parameters the model parameters
 */
void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params);

/**
 * Attach SEIR specific methods to the per-iteration FLAME GPU 2 execution structure
 * @param model the FLAME GPU Model to mutate
 */

void appendLayers(flamegpu::ModelDescription& model);

}  // namespace SEIR

}  // namespace disease
}  // namespace exateppabm
