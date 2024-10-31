#include "TimeSeriesFile.h"

#include <fmt/core.h>

namespace exateppabm {
namespace output {

TimeSeriesFile::TimeSeriesFile(std::filesystem::path path) : OutputFile(path / TimeSeriesFile::DEFAULT_FILENAME) {

}

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

    if(!this->_handle) {
        this->open();
    }

    // Print to the file handle
    fmt::print(_handle, "time,total_n,n_infected,n_infected_0_9,n_infected_10_19,n_infected_20_29,n_infected_30_39,n_infected_40_49,n_infected_50_59,n_infected_60_69,n_infected_70_79,n_infected_80\n");
    for(const auto& observation : _observations) {
        fmt::print(
            _handle,
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

    fmt::print("Timeseries data written to {}\n", std::filesystem::absolute(this->_filepath).c_str());
    return true;
}


}  // namespace output
}  // namespace exateppabm
