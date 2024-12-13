#include "exateppabm/household.h"

#include "flamegpu/flamegpu.h"
#include "exateppabm/demographics.h"
#include "exateppabm/disease.h"
#include "exateppabm/person.h"

namespace exateppabm {
namespace household {

/**
 * Agent function for person agents to emit their public information, i.e. infection status to their household
 */
FLAMEGPU_AGENT_FUNCTION(emitHouseholdStatus, flamegpu::MessageNone, flamegpu::MessageBucket) {
    // Households of 1 don't need to do any messaging, there is no one to infect
    std::uint8_t householdSize = FLAMEGPU->getVariable<std::uint8_t>(person::v::HOUSEHOLD_SIZE);
    if (householdSize > 1) {
        // output public properties to bucket message, keyed by household
        // Agent ID to avoid self messaging
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>(person::message::household_status::ID, FLAMEGPU->getVariable<flamegpu::id_t>(person::v::ID));

        // Household index
        // @todo - typedef or using statement for the household index type?
        std::uint32_t householdIdx = FLAMEGPU->getVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
        FLAMEGPU->message_out.setVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX, householdIdx);

        FLAMEGPU->message_out.setVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::
        INFECTION_STATE, FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE));
        FLAMEGPU->message_out.setVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC, FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC));
        // Set the message key, the house hold idx for bucket messaging @Todo
        FLAMEGPU->message_out.setKey(householdIdx);
    }
    return flamegpu::ALIVE;
}

/**
 * Very naive agent interaction for infection spread via house hold contact
 *
 * Agents iterate messages from their household members, potentially becoming infected
 *
 * @todo - refactor this somewhere else?
 * @todo - add per network behaviours?
 */
FLAMEGPU_AGENT_FUNCTION(interactHousehold, flamegpu::MessageBucket, flamegpu::MessageNone) {
    // Get my ID to avoid self messages
    const flamegpu::id_t id = FLAMEGPU->getVariable<flamegpu::id_t>(person::v::ID);  // FLAMEGPU->getID();

    // Get the probability of infection
    float p_s2e = FLAMEGPU->environment.getProperty<float>("p_interaction_susceptible_to_exposed");
    // Scale it for household interactions
    p_s2e *= FLAMEGPU->environment.getProperty<float>("relative_transmission_household");

    // Get my household index
    auto householdIdx = FLAMEGPU->getVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);

    // Get my age demographic
    auto demographic = FLAMEGPU->getVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);
    // Get my susceptibility modifier and modify it.
    float relativeSusceptibility = FLAMEGPU->environment.getProperty<float, demographics::AGE_COUNT>("relative_susceptibility_per_demographic", demographic);
    // Scale the probability of transmission
    p_s2e *= relativeSusceptibility;

    // Check if the current individual is susceptible to being infected
    auto infectionState = FLAMEGPU->getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);

    // @todo - this will need to change for contact tracing, the message interaction will need to occur regardless.
    if (infectionState == disease::SEIR::Susceptible) {
        // Variable to store the duration of the exposed phase (if exposed)
        float stateDuration = 0.f;

        // Iterate messages from anyone within the household
        for (const auto &message : FLAMEGPU->message_in(householdIdx)) {
            // Ignore self messages (can't infect oneself)
            if (message.getVariable<flamegpu::id_t>(person::message::household_status::ID) != id) {
                // Ignore messages from other households
                if (message.getVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX) == householdIdx) {
                    // Check if the other agent is infected
                    if (message.getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE) == disease::SEIR::InfectionState::Infected) {
                        // Roll a dice
                        float r = FLAMEGPU->random.uniform<float>();
                        if (r < p_s2e) {
                            // I have been exposed
                            infectionState = disease::SEIR::InfectionState::Exposed;
                            // Generate how long until I am infected
                            float mean = FLAMEGPU->environment.getProperty<float>("mean_time_to_infected");
                            float sd = FLAMEGPU->environment.getProperty<float>("sd_time_to_infected");
                            stateDuration = (FLAMEGPU->random.normal<float>() * sd) + mean;
                            // @todo - for now only any exposure matters. This may want to change when quantity of exposure is important?
                            // Increment the infection counter for this individual
                            FLAMEGPU->setVariable<std::uint32_t>(person::v::INFECTION_COUNT, FLAMEGPU->getVariable<std::uint32_t>(person::v::INFECTION_COUNT) + 1);
                            break;
                        }
                    }
                }
            }
        }
        // If newly exposed, store the value in global device memory.
        if (infectionState == disease::SEIR::InfectionState::Exposed) {
            FLAMEGPU->setVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE, infectionState);
            FLAMEGPU->setVariable<float>(person::v::INFECTION_STATE_DURATION, stateDuration);
        }
    }

    return flamegpu::ALIVE;
}

void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params) {
    // Define related model environment properties
    flamegpu::EnvironmentDescription env = model.Environment();

    // define the per interaction type scale factor within households
    env.newProperty<float>("relative_transmission_household", params.relative_transmission_household);

    // Get a handle to the existing person agent type, which should have already been defined.
    flamegpu::AgentDescription agent = model.Agent(person::NAME);

    // Household network variables. @todo - refactor to a separate network location?
    agent.newVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
    agent.newVariable<std::uint8_t>(person::v::HOUSEHOLD_SIZE);

    // Message list containing a persons current status for households (id, location, infection status)
    flamegpu::MessageBucket::Description householdStatusMessage = model.newMessage<flamegpu::MessageBucket>(person::message::household_status::_NAME);

    // Set the maximum bucket index to the population size. Ideally this would be household count, but that is not known at model definition time.
    // In the future this would be possible once https://github.com/FLAMEGPU/FLAMEGPU2/issues/710 is implemented
    householdStatusMessage.setUpperBound(params.n_total);

    // Add the agent id to the message.
    householdStatusMessage.newVariable<flamegpu::id_t>(person::message::household_status::ID);
    // Add the household index
    householdStatusMessage.newVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
    // Add a variable for the agent's infections status
    householdStatusMessage.newVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
    // Demographic?
    householdStatusMessage.newVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);

    // emit current status to the household
    flamegpu::AgentFunctionDescription emitHouseholdStatusDesc = agent.newFunction("emitHouseholdStatus", emitHouseholdStatus);
    emitHouseholdStatusDesc.setMessageOutput(person::message::household_status::_NAME);
    emitHouseholdStatusDesc.setMessageOutputOptional(true);
    emitHouseholdStatusDesc.setInitialState(person::states::DEFAULT);
    emitHouseholdStatusDesc.setEndState(person::states::DEFAULT);

    // Interact with other agents in the household via their messages
    flamegpu::AgentFunctionDescription interactHouseholdDesc = agent.newFunction("interactHousehold", interactHousehold);
    interactHouseholdDesc.setMessageInput(person::message::household_status::_NAME);
    interactHouseholdDesc.setInitialState(person::states::DEFAULT);
    interactHouseholdDesc.setEndState(person::states::DEFAULT);
}

void appendLayers(flamegpu::ModelDescription& model) {
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "emitHouseholdStatus");
    }
    {
        auto layer = model.newLayer();
        layer.addAgentFunction(person::NAME, "interactHousehold");
    }
}

}  // namespace household
}  // namespace exateppabm
