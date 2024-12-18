#include "TransmissionFile.h"

#include <fmt/core.h>

#include <vector>

namespace exateppabm {
namespace output {

TransmissionFile::TransmissionFile(std::filesystem::path directory) : OutputFile(directory / TransmissionFile::DEFAULT_FILENAME) { }

TransmissionFile::~TransmissionFile() { }

void TransmissionFile::reset(const std::uint32_t n_total) {
    // Reset the observations vector, with a reserved initial capacity
    this->_events = std::vector<Event>();
    this->_events.reserve(n_total);
}

void TransmissionFile::append(const Event event) {
    // @todo = ensure _events has been initialised?
    this->_events.push_back(event);
}

bool TransmissionFile::write() {
    if (!this->_handle) {
        this->open();
    }

    // Print to the file handle
    fmt::print(_handle, "id_recipient,age_group_recipient,house_no_recipient,occupation_network_recipient,transmission_event_network,time,id_source,age_group_source,house_no_source,occupation_network_source,time_infected_source,time_exposed,time_infected,time_recovered,time_susceptible\n");
    for (const auto& event : this->_events) {
        fmt::print(
            _handle,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            event.id_recipient,
            event.age_group_recipient,
            event.house_no_recipient,
            event.occupation_network_recipient,
            event.transmission_event_network,
            event.id_source,
            event.age_group_source,
            event.house_no_source,
            event.occupation_network_source,
            event.time_exposed_source,
            event.time_exposed,
            event.time_infected,
            event.time_recovered,
            event.time_susceptible);
    }

    fmt::print("Transmission File written to {}\n", std::filesystem::absolute(this->_filepath).c_str());
    return true;
}

}  // namespace output
}  // namespace exateppabm
