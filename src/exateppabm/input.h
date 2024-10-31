#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>

namespace exateppabm {

namespace input {

/** 
 * Model parameters struct, containing all parameters for a single simulation.
 */
struct config {
    /**
     * RNG seed for the simulation
     */
    std::uint64_t rng_seed = 0;
    /**
     * The parameter set id, for files with multiple parameter sets
     */
    std::uint64_t param_id = 0;
    /**
     * The duration of the simulation in days
     */
    std::uint32_t duration = 182;
    /**
     * The population size for the simulation
     */
    std::uint32_t n_total = 1024;
    /**
     * Reference size for the number of individuals within the 0_9 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_0_9 = 1;
    /**
     * Reference size for the number of individuals within the 10_19 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_10_19 = 2;
    /**
     * Reference size for the number of individuals within the 20_29 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_20_29 = 3;
    /**
     * Reference size for the number of individuals within the 30_39 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_30_39 = 4;
    /**
     * Reference size for the number of individuals within the 40_49 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_40_49 = 5;
    /**
     * Reference size for the number of individuals within the 50_59 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_50_59 = 4;
    /**
     * Reference size for the number of individuals within the 60_69 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_60_69 = 3;
    /**
     * Reference size for the number of individuals within the 70_79 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_70_79 = 2;
    /**
     * Reference size for the number of individuals within the 80 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_80 = 1;
    /**
     * The number of individuals who should be infected at the start of the simulation
     */
    std::uint32_t n_seed_infection = 1;
};

/**
 * Read simulation parameters from a CSV file
 *
 * @param p path to load parameters from
 * @return shared pointer to a configuration object 
 * @todo - support CSVs with multiple simulations, reading a single row
 */
std::shared_ptr<exateppabm::input::config> read(std::filesystem::path p);

/**
 * Print the loaded simulation configuration to stdout for validation
 *
 * @param config simulation paramater object to print
 * @todo - replace this with a method printing to disk / arbitrary file pointer, to store in the output directory
 */
void print(exateppabm::input::config config);

}  // namespace input

}  // namespace exateppabm
