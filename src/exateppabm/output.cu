#include "exateppabm/output.h"

#include <fmt/core.h>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>

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
 * Exit function to collect transmission file data for agents who are currently in a non-susceptible state, for their current infection. Some values will be -1 in this case.
 */
FLAMEGPU_EXIT_FUNCTION(transmissionFileExit) {
    fmt::print("@todo - collect transmissionData on Exit for all individuals not susceptible\n");
     // Get a handle to the person agent host api object
    auto personAgent = FLAMEGPU->agent(exateppabm::person::NAME, exateppabm::person::states::DEFAULT);
    // Get a handle to the person agent population on the host
    flamegpu::DeviceAgentVector population = personAgent.getPopulationData();
    for (const auto& person : population) {
        // If the person is not susceptible, they are in an active infection which has not yet been logged to disk.
        auto currentInfectionStatus = person.getVariable<disease::SEIR::InfectionStateUnderlyingType>(person::v::INFECTION_STATE);
        if (currentInfectionStatus != disease::SEIR::Susceptible) {
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
            // @todo - get the source workplace, household, age demo etc from disk, or do this once for all records on exit?
            // data.age_group_source = something[data.id_source - 1];
            // data.house_no_source = something[data.id_source - 1];
            // data.occupation_network_source = something[data.id_source - 1];
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

        // Add agent variables for the person agent to store information for the transmission File. These are not defined otherwise to reduce memory cost.

        // The network in which the transmission event occurred. @todo enum. 0 is home, 1 is work, 2 is random
        agent.newVariable<std::uint8_t>(person::v::TF_EVENT_NETWORK, 0);

        // The agent id for the source of infection
        agent.newVariable<flamegpu::id_t>(person::v::TF_SOURCE_ID, 0);

        // The time at which the source of infection was exposed.
        agent.newVariable<std::uint32_t>(person::v::TF_SOURCE_TIME_EXPOSED, std::numeric_limits<std::uint32_t>::max());

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
            layer.addAgentFunction(person::NAME, "transmissionFileToState");
        }
        // Host layer function which populates the transmission file data structure with info from re-susceptible individuals
        {
            auto layer = model.newLayer();
            layer.addAgentFunction(person::NAME, "transmissionFileRecordCompleted");
        }
        // Move relevant agents from transmission file back to the default state
        {
            auto layer = model.newLayer();
            layer.addAgentFunction(person::NAME, "transmissionFileToState");
        }
        // Sort agents back into their original location, for consistency with the same simulation when this file is not enabled
        {
            auto layer = model.newLayer();
            layer.addAgentFunction(person::NAME, "transmissionFileSortByID");
        }
    }
}

}  // namespace output
}  // namespace exateppabm
