#pragma once
#include "flamegpu/flamegpu.h"

#include "exateppabm/input.h"

namespace exateppabm {
namespace person {

#ifdef __CUDACC__
/**
 * Macro for allowing __host__ __device__ to be used in a header which is also included in a .cpp file, when compiling with nvcc
 */
#define DEVICE_CONSTEXPR_STRING __device__
#else  // __CUDACC__
/**
 * Macro for allowing __host__ __device__ to be used in a header which is also included in a .cpp file, when compiling with the c++ compiler
 */
#define DEVICE_CONSTEXPR_STRING
#endif  // __CUDACC__

/**
 * Agent type name for the person agents
 */
constexpr char NAME[] = "person";

/**
 * Namespace containing host device constants for state names related to the Person agent type
 */
namespace states {
constexpr char DEFAULT[] = "default";
}  // namespace states

/**
 * Namespace containing host device constant expression strings for variable names related to the person agent. To avoid repeated spelling mistakes
 */
namespace v {
// @todo - this is grim :'(. Should prolly use macros instead  Requires CUDA 11.4+ apparently.
DEVICE_CONSTEXPR_STRING constexpr char x[] = "x";
DEVICE_CONSTEXPR_STRING constexpr char y[] = "y";
DEVICE_CONSTEXPR_STRING constexpr char z[] = "z";
DEVICE_CONSTEXPR_STRING constexpr char INFECTION_STATE[] = "infection_state";
DEVICE_CONSTEXPR_STRING constexpr char INFECTION_STATE_CHANGE_DAY[] = "infection_state_change_day";
DEVICE_CONSTEXPR_STRING constexpr char INFECTION_STATE_DURATION[] = "infection_state_duration";
DEVICE_CONSTEXPR_STRING constexpr char AGE_DEMOGRAPHIC[] = "age_demographic";
}  // namespace v

/**
 * Namespace containing person-message related constants
 */
namespace message {
DEVICE_CONSTEXPR_STRING constexpr char STATUS[] = "status";
/**
 * Namespace containing variable name constants for variables in person related messages
 */
namespace v {
DEVICE_CONSTEXPR_STRING constexpr char STATUS_ID[] = "id";
}  // namespace v
}  // namespace message



/**
 * Define the agent type representing a person in the simulation, mutating the model description object.
 * @param model flamegpu2 model description object to mutate
 * @param params model parameters from parameters file
 * @param width the width of the 2D space currently used for spatials comms. to be removed once networks added.
 * @param interactionRadius spatial interaction radius for temporary infection spread behaviour. to be removed.
 */
void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params, const float width, const float interactionRadius);

/**
 * Add person related functions to the FLAMEGPU 2 layer based control flow.
 * 
 * Does not use the DAG abstraction due to previously encountered bugs with split compilation units which have not yet been pinned down / resolved.
 * 
 * @param model flamegpu2 model description object to mutate
 */
void appendLayers(flamegpu::ModelDescription& model);


// Undefine the host device macro to avoid potential macro collisions
#undef DEVICE_CONSTEXPR_STRING

}  // namespace person
}  // namespace exateppabm
