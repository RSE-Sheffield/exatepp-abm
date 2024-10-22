#include "exateppabm/output.h"

#include <cstdio>
#include <filesystem>
#include <fmt/core.h>

#include "person.h"

namespace exateppabm {

namespace output {

// Anonymous namespace for file-scoped variables used to allow data to persist between init, step and exit functions. 
// @todo - this will need making thread safe for ensemble use.
namespace {

// Filename for time series data
constexpr char TIMESERIES_FILENAME[] = "timeseries.csv";

// Path to the output directory for file output.
std::filesystem::path _outputDirectory;

// Vector of observations - i..e data for the time series csv
std::vector<exateppabm::output::observation> _observations;


}  // namespace 

/**
 * FLAME GPU init function to prepare for time series data capture throughout the simulation
 * @note - this is not sustainable for simulations with long step counts due to single output to disk, but is OK for ~365 entries. This would need refactoring to emit partial files every N iterations if finer grained data is recorded (or per agent data when that is implemented.)
 */
FLAMEGPU_INIT_FUNCTION(output_init) {
    // (re) initailise the vector of time series observations with enough points for one per simualtion step pre-allocated
    _observations = std::vector<exateppabm::output::observation>();
    _observations.reserve(FLAMEGPU->getSimulationConfig().steps);
    fmt::print("@todo - check if row 0 should be init state, or end of t0 state.\n");
}

/**
 * FLAME GPU step function, which executes at the end of each time step. 
 * Collect relevant data from agents, and store in memory for output do disk at exit.
 */
FLAMEGPU_STEP_FUNCTION(output_step) {
    auto step = FLAMEGPU->getStepCounter();
    exateppabm::output::observation observation = {};
    auto personAgent = FLAMEGPU->agent(exateppabm::person::NAME, exateppabm::person::states::DEFAULT);
    flamegpu::DeviceAgentVector population = personAgent.getPopulationData();
    // Store the iteration
    observation.time = step;
    // Store the count of people agents
    observation.total_n = population.size();
    // Perform a counting reduction over the INFECTED variable to find how many are infected
    observation.n_infected = personAgent.count<std::uint32_t>(exateppabm::person::v::INFECTED, 1u);
    // fmt::print("collecting data {}: {}/{} infected\n", step, observation.n_infected, observation.total_n);
    
    // There isn't a trivial way to perform a reduction over multiple elements of agent data at once, so for now we will iterate each agent on the host, to do a per demographic count.
    // This will likely be moved to an environment property array @todo.
    for (const auto& person : population) {
        std::uint32_t infected = person.getVariable<std::uint32_t>(exateppabm::person::v::INFECTED);
        if (infected) {
            exateppabm::person::Demographic demographicIdx = static_cast<exateppabm::person::Demographic>(person.getVariable<uint8_t>(exateppabm::person::v::DEMOGRAPHIC));
            switch(demographicIdx) {
                case exateppabm::person::Demographic::AGE_0_9:
                    observation.n_infected_0_9++;
                    break;
                case exateppabm::person::Demographic::AGE_10_19:
                    observation.n_infected_10_19++;
                    break;
                case exateppabm::person::Demographic::AGE_20_29:
                    observation.n_infected_20_29++;
                    break;
                case exateppabm::person::Demographic::AGE_30_39:
                    observation.n_infected_30_39++;
                    break;
                case exateppabm::person::Demographic::AGE_40_49:
                    observation.n_infected_40_49++;
                    break;
                case exateppabm::person::Demographic::AGE_50_59:
                    observation.n_infected_50_59++;
                    break;
                case exateppabm::person::Demographic::AGE_60_69:
                    observation.n_infected_60_69++;
                    break;
                case exateppabm::person::Demographic::AGE_70_79:
                    observation.n_infected_70_79++;
                    break;
                case exateppabm::person::Demographic::AGE_80:
                    observation.n_infected_80++;
                    break;
                default:
                    fmt::print("@todo - this should never happen\n");
                    break;
            }
        }
    }
    // Append this steps' data to the namespace-scoped data structure
    _observations.push_back(observation);
}

FLAMEGPU_EXIT_FUNCTION(output_exit) {
    // Write the time series data to disk

    // Get the filepath from the configuration via  namespace scoped member variable
    auto filepath = _outputDirectory / TIMESERIES_FILENAME;

    // @todo - refactor this into a class

    // Open the file handle

    std::FILE * fp  = {std::fopen(filepath.c_str(), "w")};
    
    // @todo - better validation / graceful handling.
    if(!fp) {
        fmt::print(stderr, "bad file handle @todo\n");
        return;
    }


    // Print to the file handle
    fmt::print(fp, "time,total_n,n_infected,n_infected_0_9,n_infected_10_19,n_infected_20_29,n_infected_30_39,n_infected_40_49,n_infected_50_59,n_infected_60_69,n_infected_70_79,n_infected_80\n");
    for(const auto& observation : _observations) {
        fmt::print(
            fp,
            "{},{},{},{},{},{},{},{},{},{},{},{}\n",
            observation.time,
            observation.total_n,
            observation.n_infected,
            observation.n_infected_0_9,
            observation.n_infected_10_19,
            observation.n_infected_20_29,
            observation.n_infected_30_39,
            observation.n_infected_40_49,
            observation.n_infected_50_59,
            observation.n_infected_60_69,
            observation.n_infected_70_79,
            observation.n_infected_80
        );
    }
    // Close the file handle.
    std::fclose(fp);
    fp = nullptr;

    fmt::print("Timeseries data written to {}\n", std::filesystem::absolute(filepath).c_str());
}

// @todo - may need to split this due to order of execution within init/step/exit funcs, if any others exist.
void define(flamegpu::ModelDescription& model, const std::filesystem::path outputDirectory) {
    // Store the output directory for access in the FLAME GPU exit function (and init?)
    // This will want refactoring for ensembles
    _outputDirectory = outputDirectory;

    // Add the init function to the model
    model.addInitFunction(output_init);
    // Add the step function to the model
    model.addStepFunction(output_step);
    // Add the exit function to the model
    model.addExitFunction(output_exit);
}

}

}  // namespace exateppabm
