#include <array>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "fmt/core.h"
#include "exateppabm/input.h"
#include "exateppabm/population.h"

/**
 * Test getReferenceMeanHouseholdSize using a number of manual, partially filled config objects.
 */
TEST(TestPopulation, getReferenceMeanHouseholdSize) {
    const double EPSILON = 1.0e-7;
    double value = 0.0;
    double expected = 0.0;
    exateppabm::input::config params = {};

    // balanced house sizes, 21 people in 6 houses
    params.household_size_1 = 1;
    params.household_size_2 = 1;
    params.household_size_3 = 1;
    params.household_size_4 = 1;
    params.household_size_5 = 1;
    params.household_size_6 = 1;
    expected = 21 / 6.0;
    value = exateppabm::population::getReferenceMeanHouseholdSize(params);
    ASSERT_NEAR(value, expected, EPSILON);

    // Just houses sizes of 1
    params.household_size_1 = 1;
    params.household_size_2 = 0;
    params.household_size_3 = 0;
    params.household_size_4 = 0;
    params.household_size_5 = 0;
    params.household_size_6 = 0;
    expected = 1.0 / 1.0;
    value = exateppabm::population::getReferenceMeanHouseholdSize(params);
    ASSERT_NEAR(value, expected, EPSILON);

    // Just houses sizes of 6
    params.household_size_1 = 0;
    params.household_size_2 = 0;
    params.household_size_3 = 0;
    params.household_size_4 = 0;
    params.household_size_5 = 0;
    params.household_size_6 = 1;
    expected = 6.0 / 1.0;
    value = exateppabm::population::getReferenceMeanHouseholdSize(params);
    ASSERT_NEAR(value, expected, EPSILON);

    // Just houses sizes of 1 or 6
    params.household_size_1 = 1;
    params.household_size_2 = 0;
    params.household_size_3 = 0;
    params.household_size_4 = 0;
    params.household_size_5 = 0;
    params.household_size_6 = 1;
    expected = 7 / 2.0;
    value = exateppabm::population::getReferenceMeanHouseholdSize(params);
    ASSERT_NEAR(value, expected, EPSILON);

    // Arbitrary mixed values
    params.household_size_1 = 1;
    params.household_size_2 = 2;
    params.household_size_3 = 3;
    params.household_size_4 = 3;
    params.household_size_5 = 2;
    params.household_size_6 = 1;
    expected = 42 / 12.0;
    value = exateppabm::population::getReferenceMeanHouseholdSize(params);
    ASSERT_NEAR(value, expected, EPSILON);
}

/**
 * Get Check the cumulative household size probability vector is as expected for sample model parameters
 */
TEST(TestPopulation, getHouseholdSizeCumulativeProbabilityVector) {
    const double EPSILON = 1.0e-7;
    std::vector<double> cumulativeP = {};
    std::vector<double> expected = {};
    constexpr std::uint32_t EXPECTED_LENGTH = 6;
    exateppabm::input::config params = {};

    // balanced house sizes equal probability of each
    params.household_size_1 = 1;
    params.household_size_2 = 1;
    params.household_size_3 = 1;
    params.household_size_4 = 1;
    params.household_size_5 = 1;
    params.household_size_6 = 1;
    expected = {{ 1/6.0, 2/6.0, 3/6.0, 4/6.0, 5/6.0, 6/6.0 }};
    cumulativeP = exateppabm::population::getHouseholdSizeCumulativeProbabilityVector(params);
    // Check the size
    ASSERT_EQ(cumulativeP.size(), EXPECTED_LENGTH);
    // Check the final element is approx equal 1.0
    ASSERT_NEAR(cumulativeP[5], 1.0, EPSILON);
    // Check each element is as expected
    for (std::size_t idx = 0; idx < cumulativeP.size(); ++idx) {
        ASSERT_NEAR(cumulativeP[idx], expected[idx], EPSILON);
    }

    // 1 or 6 only, balanced
    params.household_size_1 = 1;
    params.household_size_2 = 0;
    params.household_size_3 = 0;
    params.household_size_4 = 0;
    params.household_size_5 = 0;
    params.household_size_6 = 1;
    expected = {{ 1/2.0, 1/2.0, 1/2.0, 1/2.0, 1/2.0, 2/2.0 }};
    cumulativeP = exateppabm::population::getHouseholdSizeCumulativeProbabilityVector(params);
    // Check the size
    ASSERT_EQ(cumulativeP.size(), EXPECTED_LENGTH);
    // Check the final element is approx equal 1.0
    ASSERT_NEAR(cumulativeP[5], 1.0, EPSILON);
    // Check each element is as expected
    for (std::size_t idx = 0; idx < cumulativeP.size(); ++idx) {
        ASSERT_NEAR(cumulativeP[idx], expected[idx], EPSILON);
    }

    // imbalanced values
    params.household_size_1 = 0;
    params.household_size_2 = 1;
    params.household_size_3 = 2;
    params.household_size_4 = 3;
    params.household_size_5 = 2;
    params.household_size_6 = 1;
    expected = {{ 0/9.0, 1/9.0, 3/9.0, 6/9.0, 8/9.0, 9/9.0 }};
    cumulativeP = exateppabm::population::getHouseholdSizeCumulativeProbabilityVector(params);
    // Check the size
    ASSERT_EQ(cumulativeP.size(), EXPECTED_LENGTH);
    // Check the final element is approx equal 1.0
    ASSERT_NEAR(cumulativeP[5], 1.0, EPSILON);
    // Check each element is as expected
    for (std::size_t idx = 0; idx < cumulativeP.size(); ++idx) {
        ASSERT_NEAR(cumulativeP[idx], expected[idx], EPSILON);
    }
}

/**
 * Test that generateHouseholdStructures produces the expected data structures
 *
 *
 */
TEST(TestPopulation, generateHouseholdStructures) {
    const double EPSILON = 1.0e-7;
    std::vector<exateppabm::population::HouseholdStructure> households;
    std::mt19937_64 rng = std::mt19937_64(0);
    exateppabm::input::config params = {};
    std::uint64_t totalPeople = 0;

    // 1 house of 1 person in the 0-9 age category
    params.rng_seed = 0;
    params.n_total = 1u;
    params.household_size_1 = 1;
    params.household_size_2 = 0;
    params.household_size_3 = 0;
    params.household_size_4 = 0;
    params.household_size_5 = 0;
    params.household_size_6 = 0;
    params.population_0_9 = 1;
    params.population_10_19 = 0;
    params.population_20_29 = 0;
    params.population_30_39 = 0;
    params.population_40_49 = 0;
    params.population_50_59 = 0;
    params.population_60_69 = 0;
    params.population_70_79 = 0;
    params.population_80 = 0;
    rng.seed(params.rng_seed);
    households = exateppabm::population::generateHouseholdStructures(params, rng, false);
    // Should be 1 household
    ASSERT_EQ(households.size(), 1u);
    totalPeople = 0u;
    for (const auto & household : households) {
        totalPeople += household.size;
        // Each household should have a household size between 1 and 6 inclusive (in this case 1)
        ASSERT_EQ(household.size, 1u);
        // The total of the house size age distributions should be the same as the house size
        std::uint64_t sizePerAgeTotal = 0u;
        for (const auto & c : household.sizePerAge) {
            sizePerAgeTotal += c;
        }
        ASSERT_EQ(sizePerAgeTotal, household.size);
        // sizePerAge should be the same length as the number of age demographics
        ASSERT_EQ(household.sizePerAge.size(), exateppabm::demographics::AGE_COUNT);
        // There should also be a matching number of per-person ages
        ASSERT_EQ(household.agePerPerson.size(), household.size);
        // There should only be a person in the 0-9 age category
        ASSERT_EQ(household.sizePerAge[0], 1u);
        ASSERT_EQ(household.sizePerAge[1], 0u);
        ASSERT_EQ(household.sizePerAge[2], 0u);
        ASSERT_EQ(household.sizePerAge[3], 0u);
        ASSERT_EQ(household.sizePerAge[4], 0u);
        ASSERT_EQ(household.sizePerAge[5], 0u);
        ASSERT_EQ(household.sizePerAge[6], 0u);
        ASSERT_EQ(household.sizePerAge[7], 0u);
    }
    // Should be 1 person in total
    ASSERT_EQ(totalPeople, 1u);

    // 32 people, in houses of 1, 2 or 3 people, aged 70-79 or 80
    params.rng_seed = 0;
    params.n_total = 32u;
    params.household_size_1 = 1;
    params.household_size_2 = 1;
    params.household_size_3 = 1;
    params.household_size_4 = 0;
    params.household_size_5 = 0;
    params.household_size_6 = 0;
    params.population_0_9 = 0;
    params.population_10_19 = 0;
    params.population_20_29 = 0;
    params.population_30_39 = 0;
    params.population_40_49 = 0;
    params.population_50_59 = 0;
    params.population_60_69 = 0;
    params.population_70_79 = 1;
    params.population_80 = 1;
    rng.seed(params.rng_seed);
    households = exateppabm::population::generateHouseholdStructures(params, rng, false);
    // Should be between 1 and 32 households
    ASSERT_GE(households.size(), 1u);
    ASSERT_LE(households.size(), 32u);
    totalPeople = 0u;
    for (const auto & household : households) {
        totalPeople += household.size;
        // Each household should have a household size between 1 and 3 inclusive (in this case 1)
        ASSERT_GE(household.size, 1u);
        ASSERT_LE(household.size, 3u);
        // The total of the house size age distributions should be the same as the house size
        std::uint64_t sizePerAgeTotal = 0u;
        for (const auto & c : household.sizePerAge) {
            sizePerAgeTotal += c;
        }
        ASSERT_EQ(sizePerAgeTotal, household.size);
        // sizePerAge should be the same length as the number of age demographics
        ASSERT_EQ(household.sizePerAge.size(), exateppabm::demographics::AGE_COUNT);
        // There should also be a matching number of per-person ages
        ASSERT_EQ(household.agePerPerson.size(), household.size);
        // There should only be people in the 70-79 and 80 age categories
        ASSERT_EQ(household.sizePerAge[0], 0u);
        ASSERT_EQ(household.sizePerAge[1], 0u);
        ASSERT_EQ(household.sizePerAge[2], 0u);
        ASSERT_EQ(household.sizePerAge[3], 0u);
        ASSERT_EQ(household.sizePerAge[4], 0u);
        ASSERT_EQ(household.sizePerAge[5], 0u);
        ASSERT_LE(household.sizePerAge[6], household.size);
        ASSERT_LE(household.sizePerAge[7], household.size);
    }
    // Should be 32 person in total
    ASSERT_EQ(totalPeople, params.n_total);
}
