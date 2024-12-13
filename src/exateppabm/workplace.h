#pragma once
#include "flamegpu/flamegpu.h"

#include "exateppabm/input.h"
#include "exateppabm/demographics.h"

namespace exateppabm {
namespace workplace {

/**
 * Define flame gpu 2 model properties and functions related to workplaces
 * 
 * @param model flamegpu2 model description object to mutate
 * @param params model parameters from parameters file
 */
void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params);

/**
 * Append workplaces related agent functions to the flame gpu control flow using layers. This is intended to be called within person::appendLayers 
 *
 * Does not use the DAG abstraction due to previously encountered bugs with split compilation units which have not yet been pinned down / resolved.
 *
 * @param model flamegpu2 model description object to mutate
 */
void appendLayers(flamegpu::ModelDescription& model);


/**
 * Underlying type for workplace, to avoid excessive casting of an enum class due to FLAME GPU 2 templating
 */
typedef std::uint8_t WorkplaceUnderlyingType;

/**
 * The number of workplace networks, when a fixed number of workplace networks are used
 */
constexpr std::uint8_t WORKPLACE_COUNT = 5;

/**
 * Enum for workplace type for a person.
 * @todo - Find a nice way to make this an enum class (i.e. scoped) but still usable in FLAME GPU templated methods. Possibly implement to_underlying(Age)/from_underlying(Age)
 */
enum Workplace : WorkplaceUnderlyingType {
    WORKPLACE_SCHOOL_0_9 = 0,
    WORKPLACE_SCHOOL_10_19 = 1,
    WORKPLACE_ADULT = 2,
    WORKPLACE_70_79 = 3,
    WORKPLACE_80_PLUS = 4
};

/**
 * Generate a cumulative probability distribution adults within each workplace from generated populations
 * @param child_network_adults ratio of children to adults in the children networks
 * @param elderly_network_adults ratio of elderly to adults in the elderly networks
 * @param n_per_age number of individuals per age demographic
 * @return per-household-size cumulative probability, for sampling with a uniform distribution [0, 1)
 */
std::array<double, WORKPLACE_COUNT> getAdultWorkplaceCumulativeProbabilityArray(const double child_network_adults, const double elderly_network_adults, std::array<std::uint64_t, demographics::AGE_COUNT> n_per_age);


/**
 * For a given age demographic, assign a the workplace network given the age, per network adult cumulative probability & rng elements/state.
 * @param age age for the individual
 * @param p_adult_workplace cumulative probability for adults to be assigned to each workplace
 * @param rng RNG generator (pre-seeded)
 */

workplace::WorkplaceUnderlyingType generateWorkplaceForIndividual(const demographics::Age age, std::array<double, workplace::WORKPLACE_COUNT> p_adult_workplace, std::mt19937_64 & rng);


}  // namespace workplace
}  // namespace exateppabm
