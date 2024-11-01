#pragma once

#include "OutputFile.h"

#include <filesystem>
#include <string>
#include <vector>

#include "flamegpu/detail/SteadyClockTimer.h"

namespace exateppabm {
namespace output {

/**
 * Class for representing a time serious output file
 */
class PerformanceFile : public OutputFile {
 public:
    // Forward decl nested structs
    struct MetaData;
    struct Timers;

    /**
     * Constructor setting the path for file output.
     * @param directory parent directory for the file to be stored in (using the default filename)
     */
    explicit PerformanceFile(std::filesystem::path directory);

    /**
     * Dtor
     */
    ~PerformanceFile();

    /**
     * Reset the objects internal data structures
     */
    void reset();

    /**
     * Write the contents of the time series file to disk at the configured path
     *
     * @return bool indicating success
     */
    bool write();

    /**
     * Structure for data about the timed simulation
     */
    struct MetaData {
        /**
         * Build configuration
         */
        std::string build_type = "@todo";
        /**
         * FLAME GPU seatbelts
         */
        bool flamegpu_seatbelts = true;
        /**
         * Device ordinal
         */
        std::uint64_t device = 0;
        /**
         * Device string name
         */
        std::string device_name = "unknown";
        /**
         * Device number of SMs
         */
        unsigned int device_sm_count = 0u;
        /**
         * device memory total in bytes
         */
        size_t device_memory = 0u;
        /**
         * Absolute path to input parameters file
         */
        std::filesystem::path parameter_path;
        /**
         * Simulation duration in days
         */
        std::uint64_t duration;
        /**
         * Population size
         */
        std::uint64_t n_total;
    };


    /**
     * Collection of timing objects
     */
    struct Timers {
        /**
         * Steady clock timer for the total program
         */
        flamegpu::detail::SteadyClockTimer totalProgram;
        /**
         * Steady clock timer for config parsing
         */
        flamegpu::detail::SteadyClockTimer configParsing;
        /**
         * Steady clock timer for model construction
         */
        flamegpu::detail::SteadyClockTimer preSimulate;
        /**x
         * Steady clock timer for simulate()
         */
        flamegpu::detail::SteadyClockTimer simulate;
        /**
         * Steady clock timer for post simulation (excluding output of this file)
         */
        flamegpu::detail::SteadyClockTimer postSimulate;
        /**
         * FLAME GPU 2 simulation object reported elapsed RTC time 
         */
        double flamegpuRTCElapsed = 0.;
        /**
         * FLAME GPU 2 simulation object reported elapsed INIT function time
         */
        double flamegpuInitElapsed = 0.;
        /**
         * FLAME GPU 2 simulation object reported elapsed Exit function time
         */
        double flamegpuExitElapsed = 0.;
        /**
         * FLAME GPU 2 simulation object reported elapsed Simulation time
         */
        double flamegpuSimulateElapsed = 0.;
    };

    /**
     * Metadata about the simulation being timed
     */
    MetaData metadata;

    /**
     * Timer objects
     */
    Timers timers;

 private:
    /**
     * Default filename for output
     */
    constexpr static char DEFAULT_FILENAME[] = "performance.json";
};

}  // namespace output
}  // namespace exateppabm
