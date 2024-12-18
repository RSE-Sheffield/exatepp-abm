#include <memory>

// CLI11 command line interface
#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"

// fmt for modern c++ string formatting
#include "fmt/core.h"
#include "fmt/chrono.h"

// Flamegpu's main header
#include "flamegpu/flamegpu.h"

// exateppabm includes
#include "exateppabm/exatepp_abm.h"
#include "exateppabm/constants.h"
#include "exateppabm/typedefs.h"
#include "exateppabm/cli.h"
#include "exateppabm/demographics.h"
#include "exateppabm/disease.h"
#include "exateppabm/input.h"
#include "exateppabm/output.h"
#include "exateppabm/output/PerformanceFile.h"
#include "exateppabm/person.h"
#include "exateppabm/population.h"
#include "exateppabm/util.h"
#include "exateppabm/visualisation.h"

namespace exateppabm {

int entrypoint(int argc, char* argv[]) {
    // Build the CLI
    // @todo - should this move to main.cpp, and instead be passed to entrypoint?
    auto cli_params = std::make_shared<exateppabm::cli::params>();
    CLI::App app{"ExaTEPP Agent Based Model epidemiology demonstrator"};
    argv = app.ensure_utf8(argv);
    exateppabm::cli::setup(app, cli_params);

    // Parse the CLI
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    // print the cli
    if (cli_params->verbosity > 0) {
        exateppabm::cli::print(*cli_params);
    }
    // Prep for performance data capture (this is the earliest we know the output location for now)
    auto perfFile = exateppabm::output::PerformanceFile(cli_params->outputDir);
    perfFile.timers.totalProgram.start();
    perfFile.timers.configParsing.start();

    // Parse the provided path to an input file
    auto config = exateppabm::input::read(cli_params->inputParamFile, cli_params->inputParamLine);

    // print the parsed config.
    if (cli_params->verbosity > 0) {
        exateppabm::input::print(*config);
    } else {
        fmt::print("{} day simulation with {}/{} individuals infected at t0\n", config->duration, config->n_seed_infection, config->n_total);
    }

    perfFile.timers.configParsing.stop();
    perfFile.timers.preSimulate.start();

    // Populate performance data with config and device values
    perfFile.metadata.device_name = exateppabm::util::getGPUName(cli_params->device);
    perfFile.metadata.device_sm_count = exateppabm::util::getGPUMultiProcessorCount(cli_params->device);
    perfFile.metadata.device_memory = exateppabm::util::getGPUMemory(cli_params->device);
    perfFile.metadata.build_type = exateppabm::util::getCMakeBuildType();
    perfFile.metadata.flamegpu_seatbelts = exateppabm::util::getSeatbeltsEnabled();
    perfFile.metadata.parameter_path = cli_params->inputParamFile;
    perfFile.metadata.duration = config->duration;
    perfFile.metadata.n_total = config->n_total;

    // Initialise the CUDA context on the requested device to move timing (for now)
    exateppabm::util::initialiseCUDAContext(cli_params->device);

    // Create the flamegpu 2 model description object
    flamegpu::ModelDescription model("ExaTEPP ABM demonstrator");

    // Add the person agent to the model description
    exateppabm::person::define(model, *config);

    // Define demographic related variables
    exateppabm::demographics::define(model, *config);

    // Define disease related variables and methods
    exateppabm::disease::SEIR::define(model, *config);

    // Define init function for population generation
    exateppabm::population::define(model, *config, cli_params->verbosity > 0);

    // Add init, step and exit functions related to data collection and output. This may want refactoring when multiple output files are supported or collected data becomes more complex.
    exateppabm::output::define(model, cli_params->outputDir, cli_params->individualFile, cli_params->transmissionFile);

    // Build the model control flow. This will want abstracting more in the future @todo
    // @note - not using the DAG control flow due to bugs encountered in another project when splitting between compilation units.
    exateppabm::person::appendLayers(model);
    // Add disease progression
    exateppabm::disease::SEIR::appendLayers(model);

    // Construct the Simulation instance from the model.
    flamegpu::CUDASimulation simulation(model);

    // Setup the visualisation (if enabled)
    bool enableVisualisation = true;  // @todo make cli controllable if keeping
    bool paused = true;  // @todo make cli controllable
    unsigned int visSPS = 20;  // @todo make cli controllable. target simulation steps per second for visualisation purposes.
    exateppabm::visualisation::setup(enableVisualisation, model, simulation, paused, visSPS);

    // Setup simulation configuration options

    // If verbosity is high enough (-vvv or more) then enable flamegpu's verbose output
    if (cli_params->verbosity > 2) {
        simulation.SimulationConfig().verbosity = flamegpu::Verbosity::Verbose;
    }

    simulation.SimulationConfig().steps = config->duration;  // @todo - change this to be controlled by an exit condition?

    // Seed the FLAME GPU 2 RNG seed. This is independent from RNG on the host, but we only have one RNG engine available in FLAME GPU 2 currently.
    simulation.SimulationConfig().random_seed = config->rng_seed;  // @todo - split seeds

    // Set the GPU index
    simulation.CUDAConfig().device_id = cli_params->device;

    perfFile.timers.preSimulate.stop();

    // Run the simulation
    perfFile.timers.simulate.start();

    simulation.simulate();

    perfFile.timers.simulate.stop();
    perfFile.timers.postSimulate.start();

    perfFile.timers.flamegpuRTCElapsed = simulation.getElapsedTimeRTCInitialisation();
    perfFile.timers.flamegpuInitElapsed = simulation.getElapsedTimeInitFunctions();
    perfFile.timers.flamegpuExitElapsed = simulation.getElapsedTimeExitFunctions();
    perfFile.timers.flamegpuSimulateElapsed = simulation.getElapsedTimeSimulation();

    perfFile.timers.postSimulate.start();


    // Join the visualisation thread (if required)
    exateppabm::visualisation::join();

    perfFile.timers.postSimulate.stop();
    perfFile.timers.totalProgram.stop();
    perfFile.write();

    return EXIT_SUCCESS;
}

}  // namespace exateppabm
