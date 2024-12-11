#pragma once
#include <array>
#include <memory>
#include <random>
#include <vector>
#include "flamegpu/flamegpu.h"
#include "exateppabm/demographics.h"
#include "exateppabm/person.h"
#include "exateppabm/input.h"

namespace exateppabm {

namespace population {

/**
 * Define FLAME GPU init function to generate a population of agents for a simulation.
 * 
 * This method additionally stores several values in an anonymous namespace for use during the init function (which cannot take additional parameters)
 *
 * @param model the model the population is to be associated with (to get information about the person agent type from)
 * @param params the model parameters struct for this simulation
 * @param verbose if verbose output is enabled 
 */
void define(flamegpu::ModelDescription& model, const exateppabm::input::config params, const bool verbose);

/**
 * Get the number of agents per demographic which were initialised to be infected, for the most recent call to generate.
 * 
 * This is a workaround to make these values available in a FLAMEGPU_INIT_FUNC.
 * 
 * @todo - refactor this during generation of a realistic population of agents
 * @todo - refactor to be able to do this in a thread safe way from within an INIT function.
 * @note - 9 element std array, don't mind this creating a copy for a one time use method.
 * @return std::array containing the number of each demographic which were initialised to be infected
 */
std::array<std::uint64_t, exateppabm::demographics::AGE_COUNT> getPerDemographicInitialInfectionCount();

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
enum Workplace: WorkplaceUnderlyingType {
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

WorkplaceUnderlyingType generateWorkplaceForIndividual(const demographics::Age age, std::array<double, WORKPLACE_COUNT> p_adult_workplace, std::mt19937_64 & rng);

}  // namespace population

}  // namespace exateppabm
