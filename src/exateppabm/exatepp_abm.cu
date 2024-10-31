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
#include "exateppabm/input.h"
#include "exateppabm/output.h"
#include "exateppabm/person.h"
#include "exateppabm/population.h"
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

    // @temp - print the cli
    exateppabm::cli::print(*cli_params);

    // Parse the provided path to an input file
    auto config = exateppabm::input::read(cli_params->inputParamFile);

    // @temp - print the parsed config.
    exateppabm::input::print(*config);

    // Create the flamegpu 2 model description object
    flamegpu::ModelDescription model("ExaTEPP ABM demonstrator");

    // Add the person agent to the model description
    const float env_width = std::ceil(std::sqrt(config->n_total));
    constexpr float interactionRadius = 1.5f;
    exateppabm::person::define(model, env_width, interactionRadius);

    // Add init, step and exit functions related to data collection and output. This may want refactoring when multiple output files are supported or collected data becomes more complex.
    exateppabm::output::define(model, cli_params->outputDir);

    // Build the model control flow. This will want abstracting more in the future @todo
    // @note - not using the DAG control flow due to bugs encountered in another project when splitting between compilation units.
    exateppabm::person::appendLayers(model);

    // Construct the Simulation instance from the model.
    flamegpu::CUDASimulation simulation(model);

    // Setup the visualisation (if enabled)
    bool enableVisualisation = true;  // @todo make cli controllable if keeping
    bool paused = true;  // @todo make cli controllable
    unsigned int visSPS = 20;  // @todo make cli controllable. target simulation steps per second for visualisation purposes.
    exateppabm::visualisation::setup(enableVisualisation, model, simulation, paused, visSPS);

    // Setup simulation configuration options

    simulation.SimulationConfig().steps = config->duration;  // @todo - change this to be controlled by an exit condition?

    // Seed the FLAME GPU 2 RNG seed. This is independent from RNG on the host, but we only have one RNG engine available in FLAME GPU 2 currently.
    simulation.SimulationConfig().random_seed = config->rng_seed;  // @todo - split seeds

    // Set the GPU index
    simulation.CUDAConfig().device_id = cli_params->device;

    // Generate the population of agents.
    // @todo - this should probably be an in init function for ease of moving to a ensembles, but then cannot pass parameters in.
    const std::uint64_t pop_seed = config->rng_seed;  // @todo - split seeds
    auto personPopulation = exateppabm::population::generate(model, *config, env_width, interactionRadius);
    if (personPopulation == nullptr) {
        throw std::runtime_error("@todo - bad population generation function.");
    }
    simulation.setPopulationData(*personPopulation);

    // Run the simulation
    simulation.simulate();

    // Join the visualisation thread (if required)
    exateppabm::visualisation::join();

    fmt::print("@todo output timing data\n");

    return EXIT_SUCCESS;
}

}  // namespace exateppabm