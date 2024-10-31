#pragma once

#include "OutputFile.h"

#include <filesystem>
#include <vector>

namespace exateppabm {
namespace output {

/**
 * Class for representing a time serious output file.
 */
class TimeSeriesFile : public OutputFile {
public:

    // Forward decl nested struct
    struct Observations;

    TimeSeriesFile(std::filesystem::path path);
    ~TimeSeriesFile();
    void resetObservations(unsigned int steps);
    void appendObservations(Observations observations);
    bool write();

    struct Observations {
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

private:

    /**
     * Default filename
     */
    constexpr static char DEFAULT_FILENAME[] = "timeseries.csv";

    /**
     * Private member containing the observation data
     */
    std::vector<Observations> _observations;

};

}  // namespace output
}  // namespace exateppabm
