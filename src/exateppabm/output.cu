#include "exateppabm/output.h"

#include <cstdio>
#include <filesystem>
#include <fmt/core.h>

#include "output/OutputFile.h"
#include "output/TimeSeriesFile.h"
#include "person.h"

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
    // Perform a counting reduction over the INFECTED variable to find how many are infected
    observations.n_infected = personAgent.count<std::uint32_t>(exateppabm::person::v::INFECTED, 1u);
    // There isn't a trivial way to perform a reduction over multiple elements of agent data at once, so for now we will iterate each agent on the host, to do a per demographic count.
    // This will perform a number of D2H memory copies which harm performance
    // This will likely be moved to an environment property array @todo.
    for (const auto& person : population) {
        std::uint32_t infected = person.getVariable<std::uint32_t>(exateppabm::person::v::INFECTED);
        if (infected) {
            exateppabm::person::Demographic demographicIdx = static_cast<exateppabm::person::Demographic>(person.getVariable<uint8_t>(exateppabm::person::v::DEMOGRAPHIC));
            switch(demographicIdx) {
                case exateppabm::person::Demographic::AGE_0_9:
                    observations.n_infected_0_9++;
                    break;
                case exateppabm::person::Demographic::AGE_10_19:
                    observations.n_infected_10_19++;
                    break;
                case exateppabm::person::Demographic::AGE_20_29:
                    observations.n_infected_20_29++;
                    break;
                case exateppabm::person::Demographic::AGE_30_39:
                    observations.n_infected_30_39++;
                    break;
                case exateppabm::person::Demographic::AGE_40_49:
                    observations.n_infected_40_49++;
                    break;
                case exateppabm::person::Demographic::AGE_50_59:
                    observations.n_infected_50_59++;
                    break;
                case exateppabm::person::Demographic::AGE_60_69:
                    observations.n_infected_60_69++;
                    break;
                case exateppabm::person::Demographic::AGE_70_79:
                    observations.n_infected_70_79++;
                    break;
                case exateppabm::person::Demographic::AGE_80:
                    observations.n_infected_80++;
                    break;
                default:
                    fmt::print("@todo - this should never happen\n");
                    break;
            }
        }
    }
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

}

}  // namespace exateppabm
