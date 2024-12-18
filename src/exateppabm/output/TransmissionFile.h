#pragma once

#include "OutputFile.h"

#include <filesystem>
#include <limits>
#include <vector>

#include "flamegpu/flamegpu.h"
#include "exateppabm/demographics.h"
#include "exateppabm/person.h"

namespace exateppabm {
namespace output {

/**
 * Class for representing the transmission file, a csv containing information about each transmission event and subsequent progress of disease
 */
class TransmissionFile : public OutputFile {
 public:
    // Forward decl nested struct
    struct Event;

    /**
     * Constructor setting the path for file output.
     * @param directory parent directory for the file to be stored in (using the default filename)
     */
    explicit TransmissionFile(std::filesystem::path directory);

    /**
     * Dtor
     */
    ~TransmissionFile();

    /**
     * Reset the objects internal data structures, and pre-allocate memory for storing data for the current simulation
     * @param n_total number of people in total
     */
    void reset(std::uint32_t n_total);

    /**
     * Append data related to an infection event. This should be recorded when the infection is over (i.e. returns to susceptible) or on simulation exit, to avoid multiple entries for the same infection
     *
     * @param event data for an infection, either from time of return to susceptible, or the end of the simulation.
     */
    void append(Event event);

    /**
     * Write the contents of the time series file to disk at the configured path
     *
     * @return bool indicating success
     */
    bool write();

    /**
     * Structure for data observed at a single point in time within the time series
     */
    struct Event {
        /**
         * ID for the recipient
         */
        std::uint32_t id_recipient = 0;
        /**
         * Age group of the recipient
         */
        exateppabm::demographics::AgeUnderlyingType age_group_recipient = 0;
        /**
         * Household index of the recipient
         */
        std::uint32_t house_no_recipient = 0;
        /**
         * Work network index of the recipient
         */
        std::uint32_t occupation_network_recipient = 0;
        /**
         * The network in which the recipient was infected, essentially home, work or random
         */
        std::uint8_t transmission_event_network = 0;
        /**
         * ID for the source of infection
         */
        std::uint32_t id_source = 0;
        /**
         * Age group of the source
         */
        exateppabm::demographics::AgeUnderlyingType age_group_source = 0;
        /**
         * Household index of the source
         */
        std::uint32_t house_no_source = 0;
        /**
         * Work network index of the source
         */
        std::uint32_t occupation_network_source = 0;
        /**
         * When the source was exposed
         */
        std::uint32_t time_exposed_source = std::numeric_limits<std::uint32_t>::max();
        // @todo - migrate these into a structure in SEIR.h, if supporting multiple disease models
        /**
         * Time at which this infection event moved from susceptible to exposed, i.e. the time of transmission.
         UINT_MAX if not reached.
         */
        std::uint32_t time_exposed = std::numeric_limits<std::uint32_t>::max();
        /**
         * Time at which this infection event moved from exposed to infected. UINT_MAX if not reached
         */
        std::uint32_t time_infected = std::numeric_limits<std::uint32_t>::max();
        /**
         * Time at which this infection event moved from infected to recovered. UINT_MAX if not reached
         */
        std::uint32_t time_recovered = std::numeric_limits<std::uint32_t>::max();
        /**
         * Time at which this infection event moved from recovered to susceptible. UINT_MAX if not reached
         */
        std::uint32_t time_susceptible = std::numeric_limits<std::uint32_t>::max();
    };

 private:
    /**
     * Default filename for output
     * @todo - factor in the run index for ensembles?
     */
    constexpr static char DEFAULT_FILENAME[] = "transmission_file.csv";
    /**
     * Private member containing the observation data
     */
    std::vector<Event> _events;
};

}  // namespace output
}  // namespace exateppabm
