#pragma once

#include "OutputFile.h"

#include <filesystem>
#include <vector>

namespace exateppabm {
namespace output {

/**
 * Class for representing a time serious output file
 */
class TimeSeriesFile : public OutputFile {
 public:
    // Forward decl nested struct
    struct Observations;

    /**
     * Constructor setting the path for file output.
     * @param directory parent directory for the file to be stored in (using the default filename)
     */
    explicit TimeSeriesFile(std::filesystem::path directory);

    /**
     * Dtor
     */
    ~TimeSeriesFile();

    /**
     * Reset the objects internal data structures, and pre-allocate memory for storing data for the current simulation
     * @param steps number of time series observations to pre allocate for
     */
    void resetObservations(unsigned int steps);

    /**
     * Append a set of observations for a single time point to the internal data structure
     *
     * @param observations observations from a single time point
     */
    void appendObservations(Observations observations);

    /**
     * Write the contents of the time series file to disk at the configured path
     *
     * @return bool indicating success
     */
    bool write();

    /**
     * Structure for data observed at a single point in time within the time series
     */
    struct Observations {
        /**
         * The time for the time series observation
         */
        std::uint64_t time = 0;
        /**
         * total number of agents in the simulation, redundant per row but CSV is not ideal.
         */
        std::uint64_t total_n = 0;
        /**
         * Cumulative infected
         */
        std::uint64_t total_infected = 0;
        /**
         * Cumulative infected in the 0_9 age demographic
         */
        std::uint64_t total_infected_0_9 = 0;
        /**
         * Cumulative infected in the 10_19 age demographic
         */
        std::uint64_t total_infected_10_19 = 0;
        /**
         * Cumulative infected in the 20_29 age demographic
         */
        std::uint64_t total_infected_20_29 = 0;
        /**
         * Cumulative infected in the 30_39 age demographic
         */
        std::uint64_t total_infected_30_39 = 0;
        /**
         * Cumulative infected in the 40_49 age demographic
         */
        std::uint64_t total_infected_40_49 = 0;
        /**
         * Cumulative infected in the 50_59 age demographic
         */
        std::uint64_t total_infected_50_59 = 0;
        /**
         * Cumulative infected in the 60_69 age demographic
         */
        std::uint64_t total_infected_60_69 = 0;
        /**
         * Cumulative infected in the 70_79 age demographic
         */
        std::uint64_t total_infected_70_79 = 0;
        /**
         * Cumulative infected in the 80+ age demographic
         */
        std::uint64_t total_infected_80 = 0;
        /**
         * Current number of susceptible individuals
         */
        std::uint64_t n_susceptible = 0;
        /**
         * Current number of exposed individuals
         */
        std::uint64_t n_exposed = 0;
        /**
         * Current number of infected individuals
         */
        std::uint64_t n_infected = 0;
        /**
         * Current number of recovered individuals (currently once recovered can no longer be infected)
         */
        std::uint64_t n_recovered = 0;
    };

 private:
    /**
     * Default filename for output
     */
    constexpr static char DEFAULT_FILENAME[] = "timeseries.csv";
    /**
     * Private member containing the observation data
     */
    std::vector<Observations> _observations;
};

}  // namespace output
}  // namespace exateppabm
