#include "exateppabm/output.h"

#include <fmt/core.h>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <vector>

#include "exateppabm/output/OutputFile.h"
#include "exateppabm/output/TimeSeriesFile.h"
#include "exateppabm/output/PerIndividualFile.h"
#include "exateppabm/output/TransmissionFile.h"
#include "exateppabm/person.h"
#include "exateppabm/population.h"
#include "exateppabm/demographics.h"
#include "exateppabm/disease.h"
#include "exateppabm/household.h"
#include "exateppabm/workplace.h"


namespace exateppabm {

namespace output {

// Anonymous namespace for file-scoped variables used to allow data to persist between init, step and exit functions.
// @todo - this will need making thread safe for ensemble use.
namespace {

// Path to the output directory for file output.
std::filesystem::path _outputDirectory;

// Object representing the time series output file
std::unique_ptr<TimeSeriesFile> _timeSeriesFile = nullptr;

// Object representing the per individual file
std::unique_ptr<PerIndividualFile> _perIndividualFile = nullptr;

// Object representing the transmission file
std::unique_ptr<TransmissionFile> _transmissionFile = nullptr;

}  // namespace

/**
 * FLAME GPU init function to prepare for time series data capture throughout the simulation
 * @note - this is not sustainable for simulations with long step counts due to single output to disk, but is OK for ~365 entries. This would need refactoring to emit partial files every N iterations if finer grained data is recorded (or per agent data when that is implemented.)
 */
FLAMEGPU_INIT_FUNCTION(output_timeseries_init) {
    // (re) initialise the time series file data structure with preallocated room for the number of steps.
    _timeSeriesFile->resetObservations(FLAMEGPU->getSimulationConfig().steps);
    // Set the initial number of infected individuals per age demographic. @todo. Possibly move generation into an init method instead and do it their instead?
    auto totalInfectedPerDemographic = FLAMEGPU->environment.getMacroProperty<std::uint32_t, demographics::AGE_COUNT>("total_infected_per_demographic");
    const auto hostInitialInfectedPerDemo = exateppabm::population::getPerDemographicInitialInfectionCount();
    for (demographics::AgeUnderlyingType i = 0; i < hostInitialInfectedPerDemo.size(); i++) {
        totalInfectedPerDemographic[i] = hostInitialInfectedPerDemo[i];
    }
}

/**
 * FLAME GPU step function, which executes at the end of each time step.
 * Collect relevant data from agents, and store in memory for output do disk at exit.
 */
FLAMEGPU_STEP_FUNCTION(output_timeseries_step) {
    // get the current iteration number (0 indexed)
    auto step = FLAMEGPU->getStepCounter();
    // Get an object in which to store time series data
    exateppabm::output::TimeSeriesFile::Observations observations = {};
    // Get a handle to the person agent host api object
    auto personAgent = FLAMEGPU->agent(exateppabm::person::NAME, exateppabm::person::states::DEFAULT);
    // Get a handle to the person agent population on the host
    flamegpu::DeviceAgentVector population = personAgent.getPopulationData();
    // Store the iteration
    observations.time = step;
    // Store the count of people agents
    observations.total_n = population.size();
    // Perform a counting reduction over the INFECTION_STATE variable to find how many are in each of the states.
    // @todo - refactor this to be more generic.
    observations.n_susceptible = personAgent.count<std::uint32_t>(exateppabm::person::v::INFECTION_STATE, exateppabm::disease::SEIR::InfectionState::Susceptible);
    observations.n_exposed = personAgent.count<std::uint32_t>(exateppabm::person::v::INFECTION_STATE, exateppabm::disease::SEIR::InfectionState::Exposed);
    observations.n_infected = personAgent.count<std::uint32_t>(exateppabm::person::v::INFECTION_STATE, exateppabm::disease::SEIR::InfectionState::Infected);
    observations.n_recovered = personAgent.count<std::uint32_t>(exateppabm::person::v::INFECTION_STATE, exateppabm::disease::SEIR::InfectionState::Recovered);

    // Get the per-demographic count of cumulative infections from the macro environment property
    auto totalInfectedPerDemographic = FLAMEGPU->environment.getMacroProperty<std::uint32_t, demographics::AGE_COUNT>("total_infected_per_demographic");

    observations.total_infected_0_9 = totalInfectedPerDemographic[demographics::Age::AGE_0_9];
    observations.total_infected_10_19 = totalInfectedPerDemographic[demographics::Age::AGE_10_19];
    observations.total_infected_20_29 = totalInfectedPerDemographic[demographics::Age::AGE_20_29];
    observations.total_infected_30_39 = totalInfectedPerDemographic[demographics::Age::AGE_30_39];
    observations.total_infected_40_49 = totalInfectedPerDemographic[demographics::Age::AGE_40_49];
    observations.total_infected_50_59 = totalInfectedPerDemographic[demographics::Age::AGE_50_59];
    observations.total_infected_60_69 = totalInfectedPerDemographic[demographics::Age::AGE_60_69];
    observations.total_infected_70_79 = totalInfectedPerDemographic[demographics::Age::AGE_70_79];
    observations.total_infected_80 = totalInfectedPerDemographic[demographics::Age::AGE_80];

    // Sum the above to find the generic count.
    observations.total_infected = 0;
    for (demographics::AgeUnderlyingType i = 0; i < demographics::AGE_COUNT; i++) {
        observations.total_infected += totalInfectedPerDemographic[i];
    }

    // Append this steps' data to the namespace-scoped data structure
    _timeSeriesFile->appendObservations(observations);
}

FLAMEGPU_EXIT_FUNCTION(output_timeseries_exit) {
    // Write the time series data to disk
    // Open the file handle
    _timeSeriesFile->open();
    // Write data to the opened file
    _timeSeriesFile->write();
    // Close the file handle
    _timeSeriesFile->close();
}


FLAMEGPU_EXIT_FUNCTION(output_exit_per_individual) {
    // Collect per agent data
    // Get a handle to the person agent host api object
    auto personAgent = FLAMEGPU->agent(exateppabm::person::NAME, exateppabm::person::states::DEFAULT);
    // Get a handle to the person agent population on the host
    flamegpu::DeviceAgentVector population = personAgent.getPopulationData();
    for (const auto& person : population) {
        exateppabm::output::PerIndividualFile::Person personData = {};
        personData.id = static_cast<std::uint32_t>(person.getVariable<flamegpu::id_t>(person::v::ID));
        personData.age_group = person.getVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);
        personData.occupation_network = person.getVariable<workplace::WorkplaceUnderlyingType>(person::v::WORKPLACE_IDX);
        personData.house_no = person.getVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
        personData.infection_count = person.getVariable<std::uint32_t>(person::v::INFECTION_COUNT);
        _perIndividualFile->appendPerson(personData);
    }

    // Write the per individual data to disk
    // Open the file handle
    _perIndividualFile->open();
    // Write data to the opened file
    _perIndividualFile->write();
    // Close the file handle
    _perIndividualFile->close();
}

/**
 * FLAME GPU agent function condition, which conditionally moves individuals who need to update transmission file data (this time step) from the default to the transmissionFile state.
 *
 * To avoid duplicate entries for the same infection event, or having to lookup existing data on the host, this state change and transmission file logging occurs when agents return from being recovered to being susceptible.
 *
 * At the end of the simulation, any individuals not in the susceptible state have their partial entry added in an exit function.
 *
 * Truth-y return value for agents who have newly changed state, otherwise non-truthy.
 */
FLAMEGPU_AGENT_FUNCTION_CONDITION(transmissionFileDefaultToTransmissionFileCondition) {
    // Get the current iteration
    auto today = FLAMEGPU->getStepCounter();
    // Get the number of times the individual has been infected
    auto infectionCount = FLAMEGPU->getVariable<std::uint32_t>(person::v::INFECTION_COUNT);
    // If the individual has not been infected, return a falsey value
    if (infectionCount == 0) {
        return false;
    }
    // Get the iteration at which the agent returned from the recovered to susceptible time.
    auto timeSusceptible = FLAMEGPU->getVariable<std::uint32_t>(person::v::TIME_SUSCEPTIBLE);

    // Return a truthy value if the agent returned to the susceptible state today. TIME_SUSCEPTIBLE is the retun to susceptible, so init to uint32_max
    return today == timeSusceptible;
}

/**
 * FLAME GPU agent function, only executed by agents who have need to log data and are moving from the default to transmissionFile state.
 *
 * This function is essentially a no-op, but is required to take advantage of FLAME GPU agent states to reduce the performance impact of host-device communication.
 */
FLAMEGPU_AGENT_FUNCTION(transmissionFileDefaultToTransmissionFile, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}

/**
 * FLAME GPU host-layer function, which iterates data of person agents in the transmissionFile state, adding entries to the host transmission-file data structure
 *
 */
FLAMEGPU_HOST_FUNCTION(transmissionFileRecordCompletedInfections) {
    // Get a handle to the person agent host api object, for agents in the TRANSMISSION_FILE state
    auto personAgent = FLAMEGPU->agent(exateppabm::person::NAME, exateppabm::person::states::TRANSMISSION_FILE);
    // Get a handle to the person agent population on the host
    flamegpu::DeviceAgentVector population = personAgent.getPopulationData();
    for (const auto& person : population) {
        // All individuals in this state should be susceptible, but if they are not skip the iteration.
        // It should be safe to comment this out, in the name of performance.
        auto currentInfectionStatus = person.getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
        if (currentInfectionStatus != disease::SEIR::Susceptible) {
            assert(false);
            continue;
        }

        // Prep a data structure for transmission file entry
        exateppabm::output::TransmissionFile::Event data = {};
        // Store relevant data about the current individual (recipient)
        data.id_recipient = static_cast<std::uint32_t>(person.getVariable<flamegpu::id_t>(person::v::ID));
        data.age_group_recipient = person.getVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);
        data.house_no_recipient = person.getVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
        data.occupation_network_recipient = person.getVariable<workplace::WorkplaceUnderlyingType>(person::v::WORKPLACE_IDX);

        // Store which interaction network the event occured in. @todo - enum.
        data.transmission_event_network = person.getVariable<std::uint8_t>(person::v::TF_EVENT_NETWORK);

        // Store data about the source of the infection
        data.id_source = static_cast<std::uint32_t>(person.getVariable<flamegpu::id_t>(person::v::TF_SOURCE_ID));
        // data.age_group_source, data.house_no_source and data.occupation_network_source are all found on the host later.
        data.time_exposed_source = person.getVariable<std::uint32_t>(person::v::TF_SOURCE_TIME_EXPOSED);

        // Log the current value for the duration in each disease state
        data.time_exposed = person.getVariable<std::uint32_t>(person::v::TIME_EXPOSED);
        data.time_infected = person.getVariable<std::uint32_t>(person::v::TIME_INFECTED);
        data.time_recovered = person.getVariable<std::uint32_t>(person::v::TIME_RECOVERED);
        data.time_susceptible = person.getVariable<std::uint32_t>(person::v::TIME_SUSCEPTIBLE);

        // Append this individuals final infection event data to disk
        _transmissionFile->append(data);
    }
}

/**
 * FLAME GPU agent function to move agents from the newly infected state back to the default state, once their data has been logged for transmission files.
 *
 * This does not do much, other than change the agents state
 */
FLAMEGPU_AGENT_FUNCTION(transmissionFileTransmissionFileToDefault, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}

/**
 * Sort person agents by their ID after they have moved to and from the TransmissionFile state.
 *
 * This is to ensure RNG states are consistent for a given seed when transmission files are enabled, as RNG states do not move with the individual agents (currrently, this may be made an optional feature in a future FLAME GPU 2 release if there is enough demand.)
 */
FLAMEGPU_HOST_FUNCTION(transmissionFileSortByID) {
    // Get a handle to the person agent host api object, for agents in the default state
    auto personAgent = FLAMEGPU->agent(exateppabm::person::NAME, exateppabm::person::states::DEFAULT);
    // Sort the agents by their ID
    personAgent.sort<flamegpu::id_t>(exateppabm::person::v::ID, flamegpu::HostAgentAPI::Asc);
}


/**
 * Exit function to collect transmission file data for agents who are currently in a non-susceptible state, for their current infection. Some values will be -1 in this case.
 */
FLAMEGPU_EXIT_FUNCTION(transmissionFileExit) {
    // Get a handle to the person agent host api object
    auto personAgent = FLAMEGPU->agent(exateppabm::person::NAME, exateppabm::person::states::DEFAULT);
    // Get a handle to the person agent population on the host
    flamegpu::DeviceAgentVector population = personAgent.getPopulationData();

    // Prepare storage for the age, household index and workplace index for each individual, on the host, to reduce host device comms.
    // @todo - this could be stored from initialisation, to avoid fetching this from the device again.
    // We can assume agent ID's are contiguous, from 1, so we can store in the ID-1th element of vectors for fast storage and lookup.
    std::vector<demographics::AgeUnderlyingType> ageDemographics(population.size(), 0u);
    std::vector<std::uint32_t> householdIndices(population.size(), 0u);
    std::vector<workplace::WorkplaceUnderlyingType> workplaceIndices(population.size(), 0u);

    // Iterate the population, adding transmission events and storing data required for the next pass.
    for (const auto& person : population) {
        flamegpu::id_t id = person.getVariable<flamegpu::id_t>(person::v::ID);
        // Store the person's age, household and workplace for later use.
        ageDemographics[id - 1] = person.getVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);
        householdIndices[id - 1] = person.getVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
        workplaceIndices[id - 1] = person.getVariable<workplace::WorkplaceUnderlyingType>(person::v::WORKPLACE_IDX);

        // If the person is not susceptible, they are in an active infection which has not yet been logged to disk.
        auto currentInfectionStatus = person.getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
        if (currentInfectionStatus != disease::SEIR::Susceptible) {
            exateppabm::output::TransmissionFile::Event data = {};
            // Store relevant data about the current individual (recipient)
            data.id_recipient = static_cast<std::uint32_t>(id);
            data.age_group_recipient = person.getVariable<demographics::AgeUnderlyingType>(person::v::AGE_DEMOGRAPHIC);
            data.house_no_recipient = person.getVariable<std::uint32_t>(person::v::HOUSEHOLD_IDX);
            data.occupation_network_recipient = person.getVariable<workplace::WorkplaceUnderlyingType>(person::v::WORKPLACE_IDX);

            // Store which interaction network the event occured in. @todo - enum.
            data.transmission_event_network = person.getVariable<std::uint8_t>(person::v::TF_EVENT_NETWORK);

            // Store data about the source of the infection
            data.id_source = static_cast<std::uint32_t>(person.getVariable<flamegpu::id_t>(person::v::TF_SOURCE_ID));
            // data.age_group_source, data.house_no_source and data.occupation_network_source are all found on the host later.
            data.time_exposed_source = person.getVariable<std::uint32_t>(person::v::TF_SOURCE_TIME_EXPOSED);

            // Log the current value for the duration in each disease state
            data.time_exposed = person.getVariable<std::uint32_t>(person::v::TIME_EXPOSED);
            data.time_infected = person.getVariable<std::uint32_t>(person::v::TIME_INFECTED);
            data.time_recovered = person.getVariable<std::uint32_t>(person::v::TIME_RECOVERED);
            data.time_susceptible = person.getVariable<std::uint32_t>(person::v::TIME_SUSCEPTIBLE);

            // Append this individuals final infection event data to disk
            _transmissionFile->append(data);
        }
    }
    // Write the transmission data to disk
    // Open the file handle
    _transmissionFile->open();
    // fixup age, household and workplace
    _transmissionFile->fixSourceAgentData(ageDemographics, householdIndices, workplaceIndices);
    // Write data to the opened file
    _transmissionFile->write();
    // Close the file handle
    _transmissionFile->close();
}

// @todo - may need to split this due to order of execution within init/step/exit funcs, if any others exist.
void define(flamegpu::ModelDescription& model, const std::filesystem::path outputDirectory, const bool individualFile, const bool transmissionFile) {
    // Store the output directory for access in the FLAME GPU exit function (and init?)
    // This will want refactoring for ensembles
    _outputDirectory = outputDirectory;
    // Construct the object representing the time series file
    _timeSeriesFile = std::make_unique<TimeSeriesFile>(_outputDirectory);

    // Add the init function to the model
    model.addInitFunction(output_timeseries_init);
    // Add the step function to the model
    model.addStepFunction(output_timeseries_step);
    // Add the exit function to the model
    model.addExitFunction(output_timeseries_exit);

    // optionally prepare for per individual file output
    if (individualFile) {
        _perIndividualFile = std::make_unique<PerIndividualFile>(_outputDirectory);
        model.addExitFunction(output_exit_per_individual);
    }

    // Optionally prepare for the transmission event file, which includes mutation of the model definition, or just set variables used to disable it in device code
    flamegpu::EnvironmentDescription env = model.Environment();
    if (!transmissionFile) {
        // Define an environment property marking this feature as disabled
        env.newProperty<bool>("transmissionFileEnabled", false, true);
    } else {
        // Initialise the file-scoped transmissionFile object
        _transmissionFile = std::make_unique<TransmissionFile>(_outputDirectory);

        // Define an environment property marking this feature as enabled
        env.newProperty<bool>("transmissionFileEnabled", true, true);

        // @todo - might be nicer to move some of this to person.cu, undecided.
        // Get a handle to the person agent type
        flamegpu::AgentDescription agent = model.Agent(person::NAME);

        // Add the new agent state
        agent.newState(person::states::TRANSMISSION_FILE);

        // Agent variables for some agent variables only used for transmission file output are currently always tracked, as abstraction of these features is non trivial (must be defined in agent memory and messages)

        // Define agent function with condition to move persons to the TransmissionFile state when required
        flamegpu::AgentFunctionDescription toTransmissionFileDesc = agent.newFunction("transmissionFileDefaultToTransmissionFile", transmissionFileDefaultToTransmissionFile);
        toTransmissionFileDesc.setInitialState(person::states::DEFAULT);
        toTransmissionFileDesc.setEndState(person::states::TRANSMISSION_FILE);
        toTransmissionFileDesc.setFunctionCondition(transmissionFileDefaultToTransmissionFileCondition);

        // Define agent function to move persons from TransmissionFile state to Default
        flamegpu::AgentFunctionDescription fromTransmissionFileDesc = agent.newFunction("transmissionFileTransmissionFileToDefault", transmissionFileTransmissionFileToDefault);
        fromTransmissionFileDesc.setInitialState(person::states::TRANSMISSION_FILE);
        fromTransmissionFileDesc.setEndState(person::states::DEFAULT);

        // Add the exit function to the model, which adds data for all non-suceptible individuals for their current infection event
        model.addExitFunction(transmissionFileExit);
    }
}

void appendTransmissionFileLayers(flamegpu::ModelDescription& model) {
    // If the transmission file is enabled, add agent functions/conditions to control flow, which allows efficient collect
    if (_transmissionFile != nullptr) {
        // Move relevant agents to the state for transmission file generation
        {
            auto layer = model.newLayer();
            layer.addAgentFunction(person::NAME, "transmissionFileDefaultToTransmissionFile");
        }
        // Host layer function which populates the transmission file data structure with info from re-susceptible individuals
        {
            auto layer = model.newLayer();
            layer.addHostFunction(transmissionFileRecordCompletedInfections);
        }
        // Move relevant agents from transmission file back to the default state
        {
            auto layer = model.newLayer();
            layer.addAgentFunction(person::NAME, "transmissionFileTransmissionFileToDefault");
        }
        // Sort agents back into their original location, for consistency with the same simulation when this file is not enabled
        {
            auto layer = model.newLayer();
            layer.addHostFunction(transmissionFileSortByID);
        }
    }
}

}  // namespace output
}  // namespace exateppabm
