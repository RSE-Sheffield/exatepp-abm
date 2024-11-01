#include "TimeSeriesFile.h"

#include <fmt/core.h>

#include <vector>

namespace exateppabm {
namespace output {

TimeSeriesFile::TimeSeriesFile(std::filesystem::path directory) : OutputFile(directory / TimeSeriesFile::DEFAULT_FILENAME) { }

TimeSeriesFile::~TimeSeriesFile() { }

void TimeSeriesFile::resetObservations(const unsigned int steps) {
    // Reset the observations vector
    this->_observations = std::vector<Observations>();
    this->_observations.reserve(steps);
}

void TimeSeriesFile::appendObservations(const Observations observations) {
    // @todo = ensure _observations has been initialised?
    this->_observations.push_back(observations);
}

bool TimeSeriesFile::write() {
    if (!this->_handle) {
        this->open();
    }

    // Print to the file handle
    fmt::print(_handle, "time,total_n,total_infected,total_infected_0_9,total_infected_10_19,total_infected_20_29,total_infected_30_39,total_infected_40_49,total_infected_50_59,total_infected_60_69,total_infected_70_79,total_infected_80,n_susceptible,n_exposed,n_infected,n_recovered\n");
    for (const auto& observation : _observations) {
        fmt::print(
            _handle,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            observation.time,
            observation.total_n,
            observation.total_infected,
            observation.total_infected_0_9,
            observation.total_infected_10_19,
            observation.total_infected_20_29,
            observation.total_infected_30_39,
            observation.total_infected_40_49,
            observation.total_infected_50_59,
            observation.total_infected_60_69,
            observation.total_infected_70_79,
            observation.total_infected_80,
            observation.n_susceptible,
            observation.n_exposed,
            observation.n_infected,
            observation.n_recovered);
    }

    fmt::print("Timeseries data written to {}\n", std::filesystem::absolute(this->_filepath).c_str());
    return true;
}

}  // namespace output
}  // namespace exateppabm
