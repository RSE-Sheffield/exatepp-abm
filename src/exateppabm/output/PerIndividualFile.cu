#include "PerIndividualFile.h"

#include <fmt/core.h>

#include <vector>

namespace exateppabm {
namespace output {

PerIndividualFile::PerIndividualFile(std::filesystem::path directory) : OutputFile(directory / PerIndividualFile::DEFAULT_FILENAME) { }

PerIndividualFile::~PerIndividualFile() { }

void PerIndividualFile::reset(const std::uint32_t n_total) {
    // Reset the observations vector
    this->_perPerson = std::vector<Person>();
    this->_perPerson.reserve(n_total);
}

void PerIndividualFile::appendPerson(const Person person) {
    // @todo = ensure _perPerson has been initialised?
    this->_perPerson.push_back(person);
}

bool PerIndividualFile::write() {
    if (!this->_handle) {
        this->open();
    }

    // Print to the file handle
    fmt::print(_handle, "ID,age_group,occupation_network,house_no,infection_count\n");
    for (const auto& person : _perPerson) {
        fmt::print(
            _handle,
            "{},{},{},{},{}\n",
            person.id,
            person.age_group,
            person.occupation_network,
            person.house_no,
            person.infection_count);
    }

    fmt::print("Per individual data written to {}\n", std::filesystem::absolute(this->_filepath).c_str());
    return true;
}

}  // namespace output
}  // namespace exateppabm
