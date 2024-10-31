#pragma once
#include "flamegpu/flamegpu.h"

namespace exateppabm {
namespace person {

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
__host__ __device__ constexpr char x[] = "x";
__host__ __device__ constexpr char y[] = "y";
__host__ __device__ constexpr char z[] = "z";
__host__ __device__ constexpr char INFECTED[] = "infected";
__host__ __device__ constexpr char DEMOGRAPHIC[] = "demographic";
}  // namespace v

/**
 * Namespace containing person-message related constants
 */
namespace message {
__host__ __device__ constexpr char STATUS[] = "status";
/**
 * Namespace containing variable name constants for variables in person related messages
 */
namespace v {
__host__ __device__ constexpr char STATUS_ID[] = "id";
}  // namespace v
}  // namespace message

/**
 * Enum class for age demographics for a person.
 * @todo - Switch to a non-class enum to enable use in flamegpu agent variables without casting.
 * @todo - move to it's own file? 
 */
enum class Demographic : std::uint8_t {
    AGE_0_9 = 0,
    AGE_10_19 = 1,
    AGE_20_29 = 2,
    AGE_30_39 = 3,
    AGE_40_49 = 4,
    AGE_50_59 = 5,
    AGE_60_69 = 6,
    AGE_70_79 = 7,
    AGE_80 = 8,
};
/**
 * The number of person age demographics in the enum (C++ enums are not enumerable)
 */
constexpr std::uint8_t DEMOGRAPHIC_COUNT = 9;

/**
 * Define the agent type representing a person in the simulation, mutating the model description object.
 * @param model flamegpu2 model description object to mutate
 * @param width the width of the 2D space currently used for spatials comms. to be removed once networks added.
 * @param interactionRadius spatial interaction radius for temporary infection spread behaviour. to be removed.
 */
void define(flamegpu::ModelDescription& model, const float width, const float interactionRadius);

/**
 * Add person related functions to the FLAMEGPU 2 layer based control flow.
 * 
 * Does not use the DAG abstraction due to previously encountered bugs with split compilation units which have not yet been pinned down / resolved.
 * 
 * @param model flamegpu2 model description object to mutate
 */
void appendLayers(flamegpu::ModelDescription& model);



}  // namespace person
}  // namespace exateppabm
