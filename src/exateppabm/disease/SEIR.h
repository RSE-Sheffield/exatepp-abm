#pragma once

#include <cstdint>
#include "flamegpu/flamegpu.h"
#include "exateppabm/input.h"
#include "exateppabm/person.h"
#include "exateppabm/demographics.h"


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

/**
 * Device utility function to get an individuals current infection status from global agent memory
 *
 * Templated for due to the templated DeviceAPI object
 *  *
 * @param FLAMEGPU flame gpu device API object
 * @return individuals current infection state
 */
template<typename MsgIn, typename MsgOut>
FLAMEGPU_DEVICE_FUNCTION disease::SEIR::InfectionStateUnderlyingType getCurrentInfectionStatus(flamegpu::DeviceAPI<MsgIn, MsgOut>* FLAMEGPU) {
    return FLAMEGPU->template getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
}


/**
 * Device utility function for when an individual is exposed, moving from susceptible to exposed
 *
 * Templated for due to the templated DeviceAPI object
 *
 * @param FLAMEGPU flamegpu device api object
 * @param infectionStatus current infection status for the individual to be mutated in-place
 */
template<typename MsgIn, typename MsgOut>
FLAMEGPU_DEVICE_FUNCTION void susceptibleToExposed(flamegpu::DeviceAPI<MsgIn, MsgOut>* FLAMEGPU, disease::SEIR::InfectionStateUnderlyingType& infectionStatus) {
    // Generate how long the individual will be in the exposed for.
    float mean = FLAMEGPU->environment.template getProperty<float>("mean_time_to_infected");
    float sd = FLAMEGPU->environment.template getProperty<float>("sd_time_to_infected");
    float stateDuration = (FLAMEGPU->random.template normal<float>() * sd) + mean;

    // Update the referenced value containing the individuals current infections status, used to reduce branching within a device for loop.
    infectionStatus = disease::SEIR::InfectionState::Exposed;
    // Update individuals infection state in global agent memory
    FLAMEGPU->template setVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE, infectionStatus);
    // Update the individual infection state duration in global agent memory
    FLAMEGPU->template setVariable<float>(person::v::INFECTION_STATE_DURATION, stateDuration);

    // Increment the infection counter for this individual
    person::incrementInfectionCounter(FLAMEGPU);

    // Update the agent variable tracking when the "current" infection exposure occurred.
    FLAMEGPU->template setVariable<std::uint32_t>(person::v::TIME_EXPOSED, FLAMEGPU->getStepCounter());
}


/**
 * Function for updating agent data when moved from the Exposed to Infected state
 *
 * Templated for due to the templated DeviceAPI object
 *
 * @param FLAMEGPU flamegpu device api object
 * @param infectionStatus current infection status for the individual to be mutated in-place
 */
template<typename MsgIn, typename MsgOut>
FLAMEGPU_DEVICE_FUNCTION void exposedToInfected(flamegpu::DeviceAPI<MsgIn, MsgOut>* FLAMEGPU, disease::SEIR::InfectionStateUnderlyingType& infectionStatus) {
    std::uint32_t today = FLAMEGPU->getStepCounter();
    // Get a handle to the total_infected_per_demographic macro env property
    auto totalInfectedPerDemographic = FLAMEGPU->environment.template getMacroProperty<std::uint32_t, demographics::AGE_COUNT>("total_infected_per_demographic");

    // Get the agent's demographic
    auto demographic_idx = FLAMEGPU->template getVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);

    // Update the state
    infectionStatus = disease::SEIR::InfectionState::Infected;
    FLAMEGPU->template setVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE, infectionStatus);

    // Update the day
    FLAMEGPU->template setVariable<std::uint32_t>(person::v::INFECTION_STATE_CHANGE_DAY, today);

    // Compute how long the next state will last
    float mean = FLAMEGPU->environment.template getProperty<float>("mean_time_to_recovered");
    float sd = FLAMEGPU->environment.template getProperty<float>("sd_time_to_recovered");
    float stateDuration = (FLAMEGPU->random.template normal<float>() * sd) + mean;
    FLAMEGPU->template setVariable<float>(person::v::INFECTION_STATE_DURATION, stateDuration);

    // Atomically update the number of infected individuals for the current individual's demographics when they transition into the infection state
    totalInfectedPerDemographic[demographic_idx]++;

    // Update the agent variable tracking when the "current" infection moved from exposed to infected
    FLAMEGPU->template setVariable<std::uint32_t>(person::v::TIME_INFECTED, today);
}

/**
 * Function for updating agent data when moved from the Infected to Recovered state
 *
 * Templated for due to the templated DeviceAPI object
 *
 * @param FLAMEGPU flamegpu device api object
 * @param infectionStatus current infection status for the individual to be mutated in-place
 */
template<typename MsgIn, typename MsgOut>
FLAMEGPU_DEVICE_FUNCTION void infectedToRecovered(flamegpu::DeviceAPI<MsgIn, MsgOut>* FLAMEGPU, disease::SEIR::InfectionStateUnderlyingType& infectionStatus) {
    std::uint32_t today = FLAMEGPU->getStepCounter();

    // Update the state
    infectionStatus = disease::SEIR::InfectionState::Recovered;
    FLAMEGPU->template setVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE, infectionStatus);

    // Update the day
    FLAMEGPU->template setVariable<std::uint32_t>(person::v::INFECTION_STATE_CHANGE_DAY, today);

    // Compute how long the next state will last
    float mean = FLAMEGPU->environment.template getProperty<float>("mean_time_to_susceptible");
    float sd = FLAMEGPU->environment.template getProperty<float>("sd_time_to_susceptible");
    float stateDuration = (FLAMEGPU->random.template normal<float>() * sd) + mean;
    FLAMEGPU->template setVariable<float>(person::v::INFECTION_STATE_DURATION, stateDuration);

    // Update the agent variable tracking when the "current" infection moved from infected to recovered
    FLAMEGPU->template setVariable<std::uint32_t>(person::v::TIME_RECOVERED, today);
}

/**
 * Function for updating agent data when moved from the Recovered to Susceptible state
 *
 * Templated for due to the templated DeviceAPI object
 *
 * @param FLAMEGPU flamegpu device api object
 * @param infectionStatus current infection status for the individual to be mutated in-place

 */
template<typename MsgIn, typename MsgOut>
FLAMEGPU_DEVICE_FUNCTION void recoveredToSusceptible(flamegpu::DeviceAPI<MsgIn, MsgOut>* FLAMEGPU, disease::SEIR::InfectionStateUnderlyingType& infectionStatus) {
    std::uint32_t today = FLAMEGPU->getStepCounter();

    // Update the state
    infectionStatus = disease::SEIR::Susceptible;
    FLAMEGPU->template setVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE, infectionStatus);

    // Update the day
    FLAMEGPU->template setVariable<std::uint32_t>(person::v::INFECTION_STATE_CHANGE_DAY, today);
    float stateDuration = 0;  // susceptible doesn't have a fixed duration
    FLAMEGPU->template setVariable<float>(person::v::INFECTION_STATE_DURATION, stateDuration);

    // Update the agent variable tracking when the "current" infection moved from recovered to Susceptible
    FLAMEGPU->template setVariable<std::uint32_t>(person::v::TIME_SUSCEPTIBLE, today);
}

/**
 * Function for resetting "current" infection timing data, for use after it has been recorded
 *
 * Templated for due to the templated DeviceAPI object
 *
 * @param FLAMEGPU flamegpu device api object
 */
template<typename MsgIn, typename MsgOut>
FLAMEGPU_DEVICE_FUNCTION void resetSEIRStateTimes(flamegpu::DeviceAPI<MsgIn, MsgOut>* FLAMEGPU) {
    // Update the agent variable tracking when the "current" infection moved from Recovered to Susceptible, and the data has been logged.
    const std::uint32_t invalidTime = static_cast<std::uint32_t>(-1);
    // Don't reset susceptible
    FLAMEGPU->template setVariable<std::uint32_t>(person::v::TIME_EXPOSED, invalidTime);
    FLAMEGPU->template setVariable<std::uint32_t>(person::v::TIME_INFECTED, invalidTime);
    FLAMEGPU->template setVariable<std::uint32_t>(person::v::TIME_RECOVERED, invalidTime);
}

}  // namespace SEIR

}  // namespace disease
}  // namespace exateppabm
