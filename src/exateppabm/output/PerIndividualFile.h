#pragma once

#include "OutputFile.h"

#include <filesystem>
#include <vector>

#include "exateppabm/demographics.h"
#include "exateppabm/person.h"

namespace exateppabm {
namespace output {

/**
 * Class for representing the per individual file
 */
class PerIndividualFile : public OutputFile {
 public:
    // Forward decl nested struct
    struct Person;

    /**
     * Constructor setting the path for file output.
     * @param directory parent directory for the file to be stored in (using the default filename)
     */
    explicit PerIndividualFile(std::filesystem::path directory);

    /**
     * Dtor
     */
    ~PerIndividualFile();

    /**
     * Reset the objects internal data structures, and pre-allocate memory for storing data for the current simulation
     * @param n_total number of people in total
     */
    void reset(std::uint32_t n_total);

    /**
     * Append a set of observations for a single time point to the internal data structure
     *
     * @param observations observations from a single time point
     */
    void appendPerson(Person observations);

    /**
     * Write the contents of the time series file to disk at the configured path
     *
     * @return bool indicating success
     */
    bool write();

    /**
     * Structure for data observed at a single point in time within the time series
     */
    struct Person {
        /**
         * ID for the person
         */
        std::uint32_t id = 0;
        /**
         * Age group
         */
        exateppabm::demographics::AgeUnderlyingType age_group = 0;
        /**
         * Work network index
         */
        std::uint32_t occupation_network = 0;
        /**
         * Household index
         */
        std::uint32_t house_no = 0;
        /**
         * Cumulative number of times this individual was infected
         */
        std::uint32_t infection_count = 0;
    };

 private:
    /**
     * Default filename for output
     * @todo - factor in the run index for ensembles?
     */
    constexpr static char DEFAULT_FILENAME[] = "individual_file.csv";
    /**
     * Private member containing the observation data
     */
    std::vector<Person> _perPerson;
};

}  // namespace output
}  // namespace exateppabm
