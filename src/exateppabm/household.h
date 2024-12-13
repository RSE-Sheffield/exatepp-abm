#pragma once

#include <vector>

#include "flamegpu/flamegpu.h"
#include "exateppabm/input.h"
#include "exateppabm/demographics.h"

namespace exateppabm {
namespace household {

/**
 * Define flame gpu 2 model properties and functions related to households
 * 
 * @param model flamegpu2 model description object to mutate
 * @param params model parameters from parameters file
 */
void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params);

/**
 * Append household related agent functions to the flame gpu control flow using layers. This is intended to be called within person::appendLayers 
 *
 * Does not use the DAG abstraction due to previously encountered bugs with split compilation units which have not yet been pinned down / resolved.
 *
 * @param model flamegpu2 model description object to mutate
 */
void appendLayers(flamegpu::ModelDescription& model);



/**
 * Type definition for the type used for number of individuals within a household. 
 * @todo - move this to a more sensible location?
 */
typedef std::uint8_t HouseholdSizeType;

/**
 * Structure containing data per household (total size, size of each age demographic)
 * 
 */
struct HouseholdStructure {
    /**
     * total number of individuals
     */
    HouseholdSizeType size = 0;
    /**
     * Number of individuals in each age demographic
     * @todo - should this be a map indexed by the enum? Enums in C++17 aren't the best.
     */
    std::array<HouseholdSizeType, exateppabm::demographics::AGE_COUNT> sizePerAge = {};
    /**
     * Age demographic per member of the household
     */
    std::vector<exateppabm::demographics::Age> agePerPerson = {};
};

/**
 * Get the reference mean household size from the simulation parameters
 * @param param simulation parameters
 * @return target mean household size
 */
double getReferenceMeanHouseholdSize(const exateppabm::input::config& params);

/**
 * Generate a cumulative probability distribution for household size from reference data
 * @param params model configuration parameters
 * @return per-household-size cumulative probability, for sampling with a uniform distribution [0, 1)
 */
std::vector<double> getHouseholdSizeCumulativeProbabilityVector(const exateppabm::input::config& params);

/**
 * Generate household structures, including the number of people of each age within the household
 * 
 * This is only included in the header file for testing.
 * 
 * @todo this should try and match target household demographic reference data in some way. I.e. don't make a house of 6*0-9 year olds
 * @note this supports a maximum household size of 6, rather than allowing "6+"
 * 
 * @param config simulation parameters structure
 * @param rng RNG generator (pre-seeded)
 * @param verbose if verbose output should be produced
 * @return vector of per-household age demographic counts.
 */

std::vector<HouseholdStructure> generateHouseholdStructures(const exateppabm::input::config params, std::mt19937_64 & rng, const bool verbose);


}  // namespace household
}  // namespace exateppabm
