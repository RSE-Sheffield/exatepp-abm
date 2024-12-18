#pragma once

#include <cstdint>
#include "flamegpu/flamegpu.h"
#include "exateppabm/input.h"
#include "exateppabm/person.h"

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
 * @param current infection status for the individual to be mutated in-place
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
}

}  // namespace SEIR

}  // namespace disease
}  // namespace exateppabm
