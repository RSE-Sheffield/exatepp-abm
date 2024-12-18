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

void TransmissionFile::fixSourceAgentData(std::vector<demographics::AgeUnderlyingType> ageDemographics, std::vector<std::uint32_t> householdIndices, std::vector<workplace::WorkplaceUnderlyingType> workplaceIndices) {
    // Fixup age_group_source, house_no_source and occupation_network_source from host data (to save device memory and host device comms)
    for (auto& event : this->_events) {
        if (event.id_source > 0 && event.id_source <= ageDemographics.size()) {
            event.age_group_source = ageDemographics[event.id_source];
            event.house_no_source = householdIndices[event.id_source];
            event.occupation_network_source = workplaceIndices[event.id_source];
        }
    }
}

bool TransmissionFile::write() {
    if (!this->_handle) {
        this->open();
    }

    // Print to the file handle
    fmt::print(_handle, "id_recipient,age_group_recipient,house_no_recipient,occupation_network_recipient,transmission_event_network,id_source,age_group_source,house_no_source,occupation_network_source,time_exposed_source,time_exposed,time_infected,time_recovered,time_susceptible\n");
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
