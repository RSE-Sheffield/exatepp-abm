#pragma once

#include <cstdint>

#include "flamegpu/flamegpu.h"
#include "exateppabm/input.h"

namespace exateppabm {
namespace demographics {

/**
 * Type definition for the underlying type used for the Age demographic
 */
typedef std::uint8_t AgeUnderlyingType;

/**
 * The number of person age demographics in the enum (C++ enums are not enumerable)
 */
constexpr std::uint8_t AGE_COUNT = 9;

/**
 * Enum for age demographics for a person.
 * @todo - Find a nice way to make this an enum class (i.e. scoped) but still usable in FLAME GPU templated methods. Possibly implement to_underlying(Age)/from_underlying(Age)
 */
enum Age : AgeUnderlyingType {
    AGE_0_9 = 0,
    AGE_10_19 = 1,
    AGE_20_29 = 2,
    AGE_30_39 = 3,
    AGE_40_49 = 4,
    AGE_50_59 = 5,
    AGE_60_69 = 6,
    AGE_70_79 = 7,
    AGE_80 = 8
};

/**
 * Define the agent type representing a person in the simulation, mutating the model description object.
 * @param model flamegpu2 model description object to mutate
 * @param params model parameters from parameters file
 */
void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params);

/**
 * Get an array containing one of each age demographic enum
 * 
 * This is a workaround for the lack of reflection in c++17, used to simplify code elsewhere
 * @todo constexpr? return refernce?
 * @return std::array which is the inverse of the Age enum.
 */
std::array<demographics::Age, demographics::AGE_COUNT> getAllAgeDemographics();

/**
 * Generate a cumulative probability distribution for age demographic sampling for a given simulation configuration
 * @param params model configuration parameters
 * @return per-age demographic cumulative probability, for sampling with a uniform distribution [0, 1)
 */
std::array<float, demographics::AGE_COUNT> getAgeDemographicCumulativeProbabilityArray(const exateppabm::input::config& params);

}  // namespace demographics
}  // namespace exateppabm
