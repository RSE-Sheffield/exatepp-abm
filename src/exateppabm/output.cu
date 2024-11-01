#include "exateppabm/output.h"

#include <fmt/core.h>
#include <cstdio>
#include <filesystem>
#include <memory>

#include "exateppabm/output/OutputFile.h"
#include "exateppabm/output/TimeSeriesFile.h"
#include "exateppabm/person.h"
#include "exateppabm/population.h"
#include "exateppabm/disease.h"

namespace exateppabm {

namespace output {

// Anonymous namespace for file-scoped variables used to allow data to persist between init, step and exit functions.
// @todo - this will need making thread safe for ensemble use.
namespace {

// Path to the output directory for file output.
std::filesystem::path _outputDirectory;

// Object representing the time series output file
std::unique_ptr<TimeSeriesFile> _timeSeriesFile = nullptr;

}  // namespace

/**
 * FLAME GPU init function to prepare for time series data capture throughout the simulation
 * @note - this is not sustainable for simulations with long step counts due to single output to disk, but is OK for ~365 entries. This would need refactoring to emit partial files every N iterations if finer grained data is recorded (or per agent data when that is implemented.)
 */
FLAMEGPU_INIT_FUNCTION(output_init) {
    // (re) initialise the time series file data structure with preallocated room for the number of steps.
    _timeSeriesFile->resetObservations(FLAMEGPU->getSimulationConfig().steps);
    // Set the initial number of infected individuals per age demographic. @todo. Possibly move generation into an init method instead and do it their instead?
    auto totalInfectedPerDemographic = FLAMEGPU->environment.getMacroProperty<std::uint32_t, person::DEMOGRAPHIC_COUNT>("total_infected_per_demographic");
    const auto hostInitialInfectedPerDemo = exateppabm::population::getPerDemographicInitialInfectionCount();
    for (std::uint8_t i = 0; i < hostInitialInfectedPerDemo.size(); i++) {
        totalInfectedPerDemographic[i] = hostInitialInfectedPerDemo[i];
    }
}

/**
 * FLAME GPU step function, which executes at the end of each time step.
 * Collect relevant data from agents, and store in memory for output do disk at exit.
 */
FLAMEGPU_STEP_FUNCTION(output_step) {
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
    auto totalInfectedPerDemographic = FLAMEGPU->environment.getMacroProperty<std::uint32_t, person::DEMOGRAPHIC_COUNT>("total_infected_per_demographic");

    observations.total_infected_0_9 = totalInfectedPerDemographic[static_cast<uint8_t>(person::Demographic::AGE_0_9)];
    observations.total_infected_10_19 = totalInfectedPerDemographic[static_cast<uint8_t>(person::Demographic::AGE_10_19)];
    observations.total_infected_20_29 = totalInfectedPerDemographic[static_cast<uint8_t>(person::Demographic::AGE_20_29)];
    observations.total_infected_30_39 = totalInfectedPerDemographic[static_cast<uint8_t>(person::Demographic::AGE_30_39)];
    observations.total_infected_40_49 = totalInfectedPerDemographic[static_cast<uint8_t>(person::Demographic::AGE_40_49)];
    observations.total_infected_50_59 = totalInfectedPerDemographic[static_cast<uint8_t>(person::Demographic::AGE_50_59)];
    observations.total_infected_60_69 = totalInfectedPerDemographic[static_cast<uint8_t>(person::Demographic::AGE_60_69)];
    observations.total_infected_70_79 = totalInfectedPerDemographic[static_cast<uint8_t>(person::Demographic::AGE_70_79)];
    observations.total_infected_80 = totalInfectedPerDemographic[static_cast<uint8_t>(person::Demographic::AGE_80)];

    // Sum the above to find the generic count.
    observations.total_infected = 0;
    for (uint8_t i = 0; i < person::DEMOGRAPHIC_COUNT; i++) {
        observations.total_infected += totalInfectedPerDemographic[i];
    }

    /*
    // @todo - not currently used direct agent data iteration
    for (const auto& person : population) {
        auto infected = person.getVariable<exateppabm::disease::SEIR::InfectionStateUnderlyingType>(exateppabm::person::v::INFECTION_STATE);
    }
    */

    // Append this steps' data to the namespace-scoped data structure
    _timeSeriesFile->appendObservations(observations);
}

FLAMEGPU_EXIT_FUNCTION(output_exit) {
    // Write the time series data to disk
    // Open the file handle
    _timeSeriesFile->open();
    // Write data to the opened file
    _timeSeriesFile->write();
    // Close the file handle
    _timeSeriesFile->close();
}

// @todo - may need to split this due to order of execution within init/step/exit funcs, if any others exist.
void define(flamegpu::ModelDescription& model, const std::filesystem::path outputDirectory) {
    // Store the output directory for access in the FLAME GPU exit function (and init?)
    // This will want refactoring for ensembles
    _outputDirectory = outputDirectory;
    // Construct the object representing the time series file
    _timeSeriesFile = std::make_unique<TimeSeriesFile>(_outputDirectory);

    // Add the init function to the model
    model.addInitFunction(output_init);
    // Add the step function to the model
    model.addStepFunction(output_step);
    // Add the exit function to the model
    model.addExitFunction(output_exit);
}

}  // namespace output
}  // namespace exateppabm
