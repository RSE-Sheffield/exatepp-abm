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
 * Maximum number of random daily interactions, which must be known at compile time due to FLAME GPU array variable limitations (they are arrays not vectors)
 * @todo - make this value overrideable via CMake.
 */
#ifdef EXATEPP_ABM_MAX_RANDOM_DAILY_INTERACTIONS
constexpr std::uint32_t MAX_RANDOM_DAILY_INTERACTIONS = EXATEPP_ABM_MAX_RANDOM_DAILY_INTERACTIONS;
#else
constexpr std::uint32_t MAX_RANDOM_DAILY_INTERACTIONS = 1u;  // Declaring the variable for intellisense to not complain
#error "EXATEPP_ABM_MAX_RANDOM_DAILY_INTERACTIONS is not defined"
#endif  // EXATEPP_ABM_MAX_RANDOM_DAILY_INTERACTIONS

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
DEVICE_CONSTEXPR_STRING constexpr char ID[] = "ID";  // Custom id, as FLAMEGPU getID isn't guaranteed to start at 1.
DEVICE_CONSTEXPR_STRING constexpr char INFECTION_STATE[] = "infection_state";
DEVICE_CONSTEXPR_STRING constexpr char INFECTION_STATE_CHANGE_DAY[] = "infection_state_change_day";
DEVICE_CONSTEXPR_STRING constexpr char INFECTION_STATE_DURATION[] = "infection_state_duration";
DEVICE_CONSTEXPR_STRING constexpr char INFECTION_COUNT[] = "infection_count";
DEVICE_CONSTEXPR_STRING constexpr char AGE_DEMOGRAPHIC[] = "age_demographic";
DEVICE_CONSTEXPR_STRING constexpr char HOUSEHOLD_IDX[] = "household_idx";
DEVICE_CONSTEXPR_STRING constexpr char HOUSEHOLD_SIZE[] = "household_size";
DEVICE_CONSTEXPR_STRING constexpr char WORKPLACE_IDX[] = "workplace_idx";
DEVICE_CONSTEXPR_STRING constexpr char WORKPLACE_OUT_DEGREE[] = "workplace_out_degree";
DEVICE_CONSTEXPR_STRING constexpr char RANDOM_INTERACTION_PARTNERS[] = "random_interaction_partners";
DEVICE_CONSTEXPR_STRING constexpr char RANDOM_INTERACTION_COUNT[] = "random_interaction_count";
DEVICE_CONSTEXPR_STRING constexpr char RANDOM_INTERACTION_COUNT_TARGET[] = "random_interaction_count_target";


}  // namespace v

/**
 * Define the agent type representing a person in the simulation, mutating the model description object.
 * @param model flamegpu2 model description object to mutate
 * @param params model parameters from parameters file
 */
void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params);

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
