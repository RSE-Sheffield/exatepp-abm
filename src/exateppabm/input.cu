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

    // If the line is empty, do nothing?
    if (line.length() == 0) {
        return false;
    }

    // Find the first comma
    size_t pos = line.find(',');
    std::string valueString = pos != std::string::npos ? line.substr(0, pos) : line;
    // Shorten line.
    line = pos != std::string::npos ? line.substr(pos + 1) : "";

    // Create an istring stream and attempt to parse into value
    std::istringstream iss(valueString);
    if (iss >> value) {
        // success
        return true;
    } else {
        // failure
        // @todo - warn about the value not matching the type here? outside has better info though for the error message
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
            if (!valueFromCSVLine(line, c->n_seed_infection)) {
                throw std::runtime_error("bad value for n_seed_infection during csv parsing @todo\n");
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
            if (!valueFromCSVLine(line, c->household_size_1)) {
                throw std::runtime_error("bad value for household_size_1 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->household_size_2)) {
                throw std::runtime_error("bad value for household_size_2 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->household_size_3)) {
                throw std::runtime_error("bad value for household_size_3 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->household_size_4)) {
                throw std::runtime_error("bad value for household_size_4 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->household_size_5)) {
                throw std::runtime_error("bad value for household_size_5 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->household_size_6)) {
                throw std::runtime_error("bad value for household_size_6 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->p_interaction_susceptible_to_exposed)) {
                throw std::runtime_error("bad value for p_interaction_susceptible_to_exposed during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->mean_time_to_infected)) {
                throw std::runtime_error("bad value for mean_time_to_infected during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->sd_time_to_infected)) {
                throw std::runtime_error("bad value for sd_time_to_infected during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->mean_time_to_recovered)) {
                throw std::runtime_error("bad value for mean_time_to_recovered during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->sd_time_to_recovered)) {
                throw std::runtime_error("bad value for sd_time_to_recovered during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->mean_time_to_susceptible)) {
                throw std::runtime_error("bad value for mean_time_to_susceptible during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->sd_time_to_susceptible)) {
                throw std::runtime_error("bad value for sd_time_to_susceptible during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->relative_susceptibility_0_9)) {
                throw std::runtime_error("bad value for relative_susceptibility_0_9 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->relative_susceptibility_10_19)) {
                throw std::runtime_error("bad value for relative_susceptibility_10_19 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->relative_susceptibility_20_29)) {
                throw std::runtime_error("bad value for relative_susceptibility_20_29 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->relative_susceptibility_30_39)) {
                throw std::runtime_error("bad value for relative_susceptibility_30_39 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->relative_susceptibility_40_49)) {
                throw std::runtime_error("bad value for relative_susceptibility_40_49 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->relative_susceptibility_50_59)) {
                throw std::runtime_error("bad value for relative_susceptibility_50_59 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->relative_susceptibility_60_69)) {
                throw std::runtime_error("bad value for relative_susceptibility_60_69 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->relative_susceptibility_70_79)) {
                throw std::runtime_error("bad value for relative_susceptibility_70_79 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->relative_susceptibility_80)) {
                throw std::runtime_error("bad value for relative_susceptibility_80 during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->child_network_adults)) {
                throw std::runtime_error("bad value for child_network_adults during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->elderly_network_adults)) {
                throw std::runtime_error("bad value for elderly_network_adults during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->relative_transmission_household)) {
                throw std::runtime_error("bad value for relative_transmission_household during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->relative_transmission_occupation)) {
                throw std::runtime_error("bad value for relative_transmission_occupation during csv parsing @todo\n");
            }
            if (!valueFromCSVLine(line, c->daily_fraction_work)) {
                throw std::runtime_error("bad value for daily_fraction_work during csv parsing @todo\n");
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
    fmt::print("  rng_seed = {}\n", config.rng_seed);
    fmt::print("  param_id = {}\n", config.param_id);
    fmt::print("  duration = {}\n", config.duration);
    fmt::print("  n_total = {}\n", config.n_total);
    fmt::print("  n_seed_infection = {}\n", config.n_seed_infection);
    fmt::print("  population_0_9 = {}\n", config.population_0_9);
    fmt::print("  population_10_19 = {}\n", config.population_10_19);
    fmt::print("  population_20_29 = {}\n", config.population_20_29);
    fmt::print("  population_30_39 = {}\n", config.population_30_39);
    fmt::print("  population_40_49 = {}\n", config.population_40_49);
    fmt::print("  population_50_59 = {}\n", config.population_50_59);
    fmt::print("  population_60_69 = {}\n", config.population_60_69);
    fmt::print("  population_70_79 = {}\n", config.population_70_79);
    fmt::print("  population_80 = {}\n", config.population_80);
    fmt::print("  household_size_1 = {}\n", config.household_size_1);
    fmt::print("  household_size_2 = {}\n", config.household_size_2);
    fmt::print("  household_size_3 = {}\n", config.household_size_3);
    fmt::print("  household_size_4 = {}\n", config.household_size_4);
    fmt::print("  household_size_5 = {}\n", config.household_size_5);
    fmt::print("  household_size_6 = {}\n", config.household_size_6);
    fmt::print("  p_interaction_susceptible_to_exposed = {}\n", config.p_interaction_susceptible_to_exposed);
    fmt::print("  mean_time_to_infected = {}\n", config.mean_time_to_infected);
    fmt::print("  sd_time_to_infected = {}\n", config.sd_time_to_infected);
    fmt::print("  mean_time_to_recovered = {}\n", config.mean_time_to_recovered);
    fmt::print("  sd_time_to_recovered = {}\n", config.sd_time_to_recovered);
    fmt::print("  mean_time_to_susceptible = {}\n", config.mean_time_to_susceptible);
    fmt::print("  sd_time_to_susceptible = {}\n", config.sd_time_to_susceptible);
    fmt::print("  relative_susceptibility_0_9 = {}\n", config.relative_susceptibility_0_9);
    fmt::print("  relative_susceptibility_10_19 = {}\n", config.relative_susceptibility_10_19);
    fmt::print("  relative_susceptibility_20_29 = {}\n", config.relative_susceptibility_20_29);
    fmt::print("  relative_susceptibility_30_39 = {}\n", config.relative_susceptibility_30_39);
    fmt::print("  relative_susceptibility_40_49 = {}\n", config.relative_susceptibility_40_49);
    fmt::print("  relative_susceptibility_50_59 = {}\n", config.relative_susceptibility_50_59);
    fmt::print("  relative_susceptibility_60_69 = {}\n", config.relative_susceptibility_60_69);
    fmt::print("  relative_susceptibility_70_79 = {}\n", config.relative_susceptibility_70_79);
    fmt::print("  relative_susceptibility_80 = {}\n", config.relative_susceptibility_80);
    fmt::print("  child_network_adults = {}\n", config.child_network_adults);
    fmt::print("  elderly_network_adults = {}\n", config.elderly_network_adults);
    fmt::print("  relative_transmission_household = {}\n", config.relative_transmission_household);
    fmt::print("  relative_transmission_occupation = {}\n", config.relative_transmission_occupation);
    fmt::print("  daily_fraction_work = {}\n", config.daily_fraction_work);
    fmt::print("}}\n");
}

}  // namespace input
}  // namespace exateppabm
