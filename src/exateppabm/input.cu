#include "exateppabm/input.h"

#include <fmt/core.h>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace exateppabm {
namespace input {
namespace {

/**
 * Get the next value from a csv row as a specified type
 *
 */
template<typename T>
bool valueFromCSVLine(std::string& line, T& value) {
    // trim whitespace
    line.erase(0, line.find_first_not_of(' '));
    line.erase(line.find_last_not_of(' ') + 1);
    std::istringstream iss(line);
    if (iss >> value) {
        char comma;
        iss >> comma;
        if (comma != ',') {
            // If the next character is not a comma, we've reached the end
            line.clear();
            return true;
        }
        // Remove the consumed part from the string
        size_t pos = line.find(',');
        line = line.substr(pos + 1);
        return true;
    } else {
        return false;
    }
}

}  // anon namespace

std::shared_ptr<exateppabm::input::config> read(const std::filesystem::path p, const int lineNumber) {
    // Construct the model parameters / config  struct
    auto c = std::make_shared<config>();

    bool valid_input_file = true;

    // if the path is not an empty path
    if (p != "") {
        // if the provided path is not a file, error
        if (!std::filesystem::is_regular_file(p)) {
            throw std::runtime_error("Bad input path @todo nicer error message");
        }
        // Open the file. Error if it failed to open
        std::string line;
        std::ifstream fs(p);

        if (!fs.is_open()) {
            throw std::runtime_error("failed to open parameter/config file @todo nicer error message");
        }

        // @todo - this is incredibly brittle. Using a library and respecting the column names or indices would be better. Or something actually readable like a yaml/toml file (or support both)
        // @todo - abstract this to it's own location.

        // @todo - check the header row is as expected?
        // For now discard the header row
        if (!std::getline(fs, line)) {
            throw std::runtime_error("failed to read the header line @todo nicer error message");
        }

        // Discard rows until the line number is the target line number
        for (int currentLine = 1; currentLine < lineNumber; currentLine++) {
            if (!std::getline(fs, line)) {
                throw std::runtime_error("Bad parameters file lineNumber @todo nicer errors");
            }
        }

        // Read the next line of the file which should contains the parameter values
        if (std::getline(fs, line)) {
            // Extract values from the line in the expected order, into the params struct
            // Error if any were missing / bad
            // This is brittle.
            if (!valueFromCSVLine(line, c->rng_seed)) {
                throw std::runtime_error("bad value for rng_seed during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->param_id)) {
                throw std::runtime_error("bad value for param_id during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->duration)) {
                throw std::runtime_error("bad value for duration during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->n_total)) {
                throw std::runtime_error("bad value for n_total during csv parsing @todo\n");
            }

            if (!valueFromCSVLine(line, c->population_0_9)) {
                throw std::runtime_error("bad value for population_0_9 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->population_10_19)) {
                throw std::runtime_error("bad value for population_10_19 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->population_20_29)) {
                throw std::runtime_error("bad value for population_20_29 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->population_30_39)) {
                throw std::runtime_error("bad value for population_30_39 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->population_40_49)) {
                throw std::runtime_error("bad value for population_40_49 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->population_50_59)) {
                throw std::runtime_error("bad value for population_50_59 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->population_60_69)) {
                throw std::runtime_error("bad value for population_60_69 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->population_70_79)) {
                throw std::runtime_error("bad value for population_70_79 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->population_80)) {
                throw std::runtime_error("bad value for population_80 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->n_seed_infection)) {
                throw std::runtime_error("bad value for n_seed_infection during csv parsing @todo\n");
            }

        } else {
            throw std::runtime_error("failed to read the paramameter value line @todo nicer error message");
        }

        fs.close();
        if (!valid_input_file) {
            throw std::runtime_error("Bad input file @todo nicer error message");
        }
    }
    // Return the default or populated configuration object
    return c;
}

void print(exateppabm::input::config config) {
    fmt::print("config {{\n");
    fmt::print(" rng_seed = {}\n", config.rng_seed);
    fmt::print(" param_id = {}\n", config.param_id);
    fmt::print(" duration = {}\n", config.duration);
    fmt::print(" n_total = {}\n", config.n_total);
    fmt::print(" population_0_9 = {}\n", config.population_0_9);
    fmt::print(" population_10_19 = {}\n", config.population_10_19);
    fmt::print(" population_20_29 = {}\n", config.population_20_29);
    fmt::print(" population_30_39 = {}\n", config.population_30_39);
    fmt::print(" population_40_49 = {}\n", config.population_40_49);
    fmt::print(" population_50_59 = {}\n", config.population_50_59);
    fmt::print(" population_60_69 = {}\n", config.population_60_69);
    fmt::print(" population_70_79 = {}\n", config.population_70_79);
    fmt::print(" population_80 = {}\n", config.population_80);
    fmt::print(" n_seed_infection = {}\n", config.n_seed_infection);
    fmt::print("}}\n");
}

}  // namespace input
}  // namespace exateppabm
