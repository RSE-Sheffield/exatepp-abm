#pragma once

#include <cstdint>

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

}  // namespace demographics
}  // namespace exateppabm
