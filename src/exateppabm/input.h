#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>

namespace exateppabm {

namespace input {

/**
 * Model parameters struct, containing all parameters for a single simulation.
 */
struct config {
    /**
     * RNG seed for the simulation
     */
    std::uint64_t rng_seed = 0;
    /**
     * The parameter set id, for files with multiple parameter sets
     */
    std::uint64_t param_id = 0;
    /**
     * The duration of the simulation in days
     */
    std::uint32_t duration = 364;
    /**
     * The population size for the simulation
     */
    std::uint32_t n_total = 1024;
    /**
     * The number of individuals who should be infected at the start of the simulation
     */
    std::uint32_t n_seed_infection = 1;
    /**
     * Reference size for the number of individuals within the 0_9 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_0_9 = 1;
    /**
     * Reference size for the number of individuals within the 10_19 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_10_19 = 2;
    /**
     * Reference size for the number of individuals within the 20_29 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_20_29 = 3;
    /**
     * Reference size for the number of individuals within the 30_39 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_30_39 = 4;
    /**
     * Reference size for the number of individuals within the 40_49 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_40_49 = 5;
    /**
     * Reference size for the number of individuals within the 50_59 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_50_59 = 4;
    /**
     * Reference size for the number of individuals within the 60_69 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_60_69 = 3;
    /**
     * Reference size for the number of individuals within the 70_79 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_70_79 = 2;
    /**
     * Reference size for the number of individuals within the 80 age demographic, used to compute ratios for population initialisation
     */
    std::uint64_t population_80 = 1;
    /**
     * Reference size for the number of households with 1 person, used for household generation
     */
    std::uint64_t household_size_1 = 1;
    /**
     * Reference size for the number of households with 2 people, used for household generation
     */
    std::uint64_t household_size_2 = 1;
    /**
     * Reference size for the number of households with 3 people, used for household generation
     */
    std::uint64_t household_size_3 = 1;
    /**
     * Reference size for the number of households with 4 people, used for household generation
     */
    std::uint64_t household_size_4 = 1;
    /**
     * Reference size for the number of households with 5 people, used for household generation
     */
    std::uint64_t household_size_5 = 1;
    /**
     * Reference size for the number of households with 6+ people, used for household generation. 6+ chosen as largest band based on ONS estimates https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/families/datasets/householdsbyhouseholdsizeregionsofenglandandgbconstituentcountries
     */
    std::uint64_t household_size_6 = 1;
    /**
     * The probability of an interaction between an infected individual and a susceptible individual resulting in an infection, prior to any susceptibility modifier.
     * This is not directly based on a parameter in the reference model
     * The default value is arbitrary
     */
    float p_interaction_susceptible_to_exposed = 0.2f;
    /**
     * The mean time in days from exposed to infected state
     * Default value is arbitrary
     * @todo - might be equivalent to mean_time_to_syptoms, depending on how asym works in the reference model.
     */
    float mean_time_to_infected = 4;
    /**
     * Standard deviation of time from exposed to infected state
     * Default value is arbitrary
     */
    float sd_time_to_infected = 1;
    /**
     * The mean time in days from infected to recovered state
     * Default value is arbitrary
     */
    float mean_time_to_recovered = 7;
    /**
     * Standard deviation of time from infected to recovered state
     * Default value is arbitrary
     */
    float sd_time_to_recovered = 1;
    /**
     * The mean time in days from recovered to susceptible
     * Default value is arbitrary
     */
    float mean_time_to_susceptible = 60;
    /**
     * Standard deviation of time from recovered to susceptible
     * Default value is arbitrary
     */
    float sd_time_to_susceptible = 1;
    /**
     * Relative susceptibility to infection/transmission for individuals within the 0_9 age demographic.
     * Arbitrary default value
     */
    float relative_susceptibility_0_9 = 1.0;
    /**
     * Relative susceptibility to infection/transmission for individuals within the 10_19 age demographic.
     * Arbitrary default value
     */
    float relative_susceptibility_10_19 = 1.0;
    /**
     * Relative susceptibility to infection/transmission for individuals within the 20_29 age demographic.
     * Arbitrary default value
     */
    float relative_susceptibility_20_29 = 1.0;
    /**
     * Relative susceptibility to infection/transmission for individuals within the 30_39 age demographic.
     * Arbitrary default value
     */
    float relative_susceptibility_30_39 = 1.0;
    /**
     * Relative susceptibility to infection/transmission for individuals within the 40_49 age demographic.
     * Arbitrary default value
     */
    float relative_susceptibility_40_49 = 1.0;
    /**
     * Relative susceptibility to infection/transmission for individuals within the 50_59 age demographic.
     * Arbitrary default value
     */
    float relative_susceptibility_50_59 = 1.0;
    /**
     * Relative susceptibility to infection/transmission for individuals within the 60_69 age demographic.
     * Arbitrary default value
     */
    float relative_susceptibility_60_69 = 1.0;
    /**
     * Relative susceptibility to infection/transmission for individuals within the 70_79 age demographic.
     * Arbitrary default value
     */
    float relative_susceptibility_70_79 = 1.0;
    /**
     * Relative susceptibility to infection/transmission for individuals within the 80 age demographic.
     * Arbitrary default value
     */
    float relative_susceptibility_80 = 1.0;
    /**
     * Proportion of adults to children in school networks (0-19)
     */
    double child_network_adults = 0.2;
    /**
     * Proportion of adults to elderly in work networks for retired/elderly individuals (70+)
     */
    double elderly_network_adults = 0.2;
    /**
     * Relative transmission rate for interactions within the household
     * Arbitrary default value
     */
    float relative_transmission_household = 2.0f;
    /**
     * Relative transmission rate for interactions within the occupation network 
     * Arbitrary default value
     */
    float relative_transmission_occupation = 1.0f;
    /**
     * Fraction of people in work network interacted with per day (by rng sampling)
     * 
     * @todo - this probably needs using differntly with more realistic networks
     */
    float daily_fraction_work = 0.5f;
};

/**
 * Read simulation parameters from a CSV file
 *
 * @param p path to load parameters from
 * @param lineNumber the line of CSV to use.
 * @return shared pointer to a configuration object
 * @todo - support CSVs with multiple simulations, reading a single row
 */
std::shared_ptr<exateppabm::input::config> read(std::filesystem::path p, int lineNumber);

/**
 * Print the loaded simulation configuration to stdout for validation
 *
 * @param config simulation paramater object to print
 * @todo - replace this with a method printing to disk / arbitrary file pointer, to store in the output directory
 */
void print(exateppabm::input::config config);

}  // namespace input

}  // namespace exateppabm
