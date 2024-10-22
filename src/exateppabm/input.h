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
    std::uint64_t rng_seed = 0;
    std::uint64_t param_id = 0;
    std::uint32_t duration = 182;
    std::uint32_t n_total = 1024;
    std::uint64_t population_0_9 = 1;
    std::uint64_t population_10_19 = 2;
    std::uint64_t population_20_29 = 3;
    std::uint64_t population_30_39 = 4;
    std::uint64_t population_40_49 = 5;
    std::uint64_t population_50_59 = 4;
    std::uint64_t population_60_69 = 3;
    std::uint64_t population_70_79 = 2;
    std::uint64_t population_80 = 1;
    std::uint32_t n_seed_infection = 1;
};

std::shared_ptr<exateppabm::input::config> read(std::filesystem::path p);

void print(exateppabm::input::config);

}  // namespace input

}  // namespace exateppabm
