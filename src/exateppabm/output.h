#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <memory>

#include "flamegpu/flamegpu.h"

namespace exateppabm {

namespace output {

struct observation {
    // Number representing the time point, currently day
    std::uint64_t time = 0;
    // total number of agents in the simulation, redundant per row but CSV is not ideal.
    std::uint64_t total_n = 0;  
    // // cumulative totals infected
    // std::uint64_t total_infected = 0;
    // std::uint64_t total_infected_0_9 = 0;
    // std::uint64_t total_infected_10_19 = 0;
    // std::uint64_t total_infected_20_29 = 0;
    // std::uint64_t total_infected_30_39 = 0;
    // std::uint64_t total_infected_40_49 = 0;
    // std::uint64_t total_infected_50_59 = 0;
    // std::uint64_t total_infected_60_69 = 0;
    // std::uint64_t total_infected_70_79 = 0;
    // std::uint64_t total_infected_80 = 0;
     // current infected
    std::uint64_t n_infected = 0;
    std::uint64_t n_infected_0_9 = 0;
    std::uint64_t n_infected_10_19 = 0;
    std::uint64_t n_infected_20_29 = 0;
    std::uint64_t n_infected_30_39 = 0;
    std::uint64_t n_infected_40_49 = 0;
    std::uint64_t n_infected_50_59 = 0;
    std::uint64_t n_infected_60_69 = 0;
    std::uint64_t n_infected_70_79 = 0;
    std::uint64_t n_infected_80 = 0;
};

/**
 * Define FLAME GPU model functions related to output of data to disk. 
 * 
 * I.e. Step and Exit functions (maybe host layer) which collect data through the simulation and output it to disk.
 * 
 * @todo this will likely find a better home once multiple output types are added.
 * 
 * @param model flamegpu2 model description object to mutate
 * @param outputDirectory path to directory for file output
 */
void define(flamegpu::ModelDescription& model, std::filesystem::path outputDirectory);

}

}  // namespace exateppabm
