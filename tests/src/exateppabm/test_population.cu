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


/**
 * Test getAdultWorkplaceCumulativeProbabilityArray
 */

TEST(TestPopulation, getAdultWorkplaceCumulativeProbabilityArray) {
const double EPSILON = 1.0e-7;
double child_network_adults = 0;
double elderly_network_adults = 0;
std::array<std::uint64_t, exateppabm::demographics::AGE_COUNT> n_per_age = {};
std::array<double, exateppabm::population::WORKPLACE_COUNT> p = {};
std::array<double, exateppabm::population::WORKPLACE_COUNT> expected = {};


// With 0 adults in the child or elderly networks, should all be assigned to the adult network, regardless of population counts
child_network_adults = 0;
elderly_network_adults = 0;
n_per_age = {{10, 10, 10, 10, 10, 10, 10, 10, 10}};
expected = {{0.0, 0.0, 1.0, 1.0, 1.0}};
p = exateppabm::population::getAdultWorkplaceCumulativeProbabilityArray(child_network_adults, elderly_network_adults, n_per_age);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_SCHOOL_0_9], expected[exateppabm::population::WORKPLACE_SCHOOL_0_9], EPSILON);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_SCHOOL_10_19], expected[exateppabm::population::WORKPLACE_SCHOOL_10_19], EPSILON);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_ADULT], expected[exateppabm::population::WORKPLACE_ADULT], EPSILON);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_70_79], expected[exateppabm::population::WORKPLACE_70_79], EPSILON);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_80_PLUS], expected[exateppabm::population::WORKPLACE_80_PLUS], EPSILON);

// With 1 adults per member of work networks, and a population which will support this, generate probabilities that would lead to 20% of adults being assigned to each network
child_network_adults = 1.0;
elderly_network_adults = 1.0;
// I.e. 20 children/elderly per network and 100 adults total, 20 in each network in the end
n_per_age = {{20, 20, 20, 20, 20, 20, 20, 20, 20}};
expected = {{0.2, 0.4, 0.6, 0.8, 1.0}};
p = exateppabm::population::getAdultWorkplaceCumulativeProbabilityArray(child_network_adults, elderly_network_adults, n_per_age);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_SCHOOL_0_9], expected[exateppabm::population::WORKPLACE_SCHOOL_0_9], EPSILON);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_SCHOOL_10_19], expected[exateppabm::population::WORKPLACE_SCHOOL_10_19], EPSILON);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_ADULT], expected[exateppabm::population::WORKPLACE_ADULT], EPSILON);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_70_79], expected[exateppabm::population::WORKPLACE_70_79], EPSILON);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_80_PLUS], expected[exateppabm::population::WORKPLACE_80_PLUS], EPSILON);

// With 10% in child networks, and 20% in elderly networks, and 100 adults total, generate the expected (unbalanced) values
child_network_adults = 0.1;
elderly_network_adults = 0.2;
n_per_age = {{20, 20, 0, 0, 20, 0, 0, 20, 10}};
// 2/20 adult, 2/20 adults, 10/20 adults, 4/20 adults, 2/20 adults, but cumulative.
expected = {{0.1, 0.2, 0.7, 0.9, 1.0}};
p = exateppabm::population::getAdultWorkplaceCumulativeProbabilityArray(child_network_adults, elderly_network_adults, n_per_age);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_SCHOOL_0_9], expected[exateppabm::population::WORKPLACE_SCHOOL_0_9], EPSILON);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_SCHOOL_10_19], expected[exateppabm::population::WORKPLACE_SCHOOL_10_19], EPSILON);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_ADULT], expected[exateppabm::population::WORKPLACE_ADULT], EPSILON);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_70_79], expected[exateppabm::population::WORKPLACE_70_79], EPSILON);
EXPECT_NEAR(p[exateppabm::population::WORKPLACE_80_PLUS], expected[exateppabm::population::WORKPLACE_80_PLUS], EPSILON);

// @todo - expand test with more edge case coverage
}

/**
 * Test generateWorkplaceForIndividual
 */
TEST(TestPopulation, generateWorkplaceForIndividual) {
// number of individuals to generate, to do a large enough sample to make sure returned values are in-range.
constexpr std::uint64_t n_samples = 1000u;
constexpr std::uint64_t n_samples_adult = 10000u;
// rng state
std::mt19937_64 rng(0u);
// probability array
std::array<double, exateppabm::population::WORKPLACE_COUNT> p_adult_workplace = {{0.1, 0.2, 0.8, 0.9, 1.0}};
// Expected/target counts for adults (non-cumulative)
std::array<double, exateppabm::population::WORKPLACE_COUNT> target_adult_workplace = {{
    0.1 * n_samples_adult,
    0.1 * n_samples_adult,
    0.6 * n_samples_adult,
    0.1 * n_samples_adult,
    0.1 * n_samples_adult}};
double TARGET_ADULT_EPSILON = 0.01 * n_samples_adult;

// Repeatedly generate individuals of each age band from a given probability distribution, checking they are always the expected values

// 0-9 always in their workplace
for (std::uint64_t idx = 0; idx < n_samples; ++idx) {
    auto w = exateppabm::population::generateWorkplaceForIndividual(exateppabm::demographics::AGE_0_9, p_adult_workplace, rng);
    EXPECT_EQ(w, exateppabm::population::Workplace::WORKPLACE_SCHOOL_0_9);
}
// 10-19 always in their workplace
for (std::uint64_t idx = 0; idx < n_samples; ++idx) {
    auto w = exateppabm::population::generateWorkplaceForIndividual(exateppabm::demographics::AGE_10_19, p_adult_workplace, rng);
    EXPECT_EQ(w, exateppabm::population::Workplace::WORKPLACE_SCHOOL_10_19);
}
// 70-79 always in their workplace
for (std::uint64_t idx = 0; idx < n_samples; ++idx) {
    auto w = exateppabm::population::generateWorkplaceForIndividual(exateppabm::demographics::AGE_70_79, p_adult_workplace, rng);
    EXPECT_EQ(w, exateppabm::population::Workplace::WORKPLACE_70_79);
}
// 80+ always in their workplace
for (std::uint64_t idx = 0; idx < n_samples; ++idx) {
    auto w = exateppabm::population::generateWorkplaceForIndividual(exateppabm::demographics::AGE_80, p_adult_workplace, rng);
    EXPECT_EQ(w, exateppabm::population::Workplace::WORKPLACE_80_PLUS);
}
// Adults are randomly assigned subject to the cumulative probability distribution.
std::array<std::uint64_t, exateppabm::population::WORKPLACE_COUNT> workplaceAdults = {0};
for (std::uint64_t idx = 0; idx < n_samples_adult; ++idx) {
    // Just check a single age demo for simlicity
    auto w = exateppabm::population::generateWorkplaceForIndividual(exateppabm::demographics::AGE_40_49, p_adult_workplace, rng);
    // Ensure in range
    EXPECT_GE(w, 0);
    EXPECT_LT(w, exateppabm::population::WORKPLACE_COUNT);
    // Increment the counter
    ++workplaceAdults[w];
}
// Check that each counter is roughly correct.
EXPECT_NEAR(workplaceAdults[exateppabm::population::Workplace::WORKPLACE_SCHOOL_0_9], target_adult_workplace[exateppabm::population::Workplace::WORKPLACE_SCHOOL_0_9], TARGET_ADULT_EPSILON);
EXPECT_NEAR(workplaceAdults[exateppabm::population::Workplace::WORKPLACE_SCHOOL_10_19], target_adult_workplace[exateppabm::population::Workplace::WORKPLACE_SCHOOL_10_19], TARGET_ADULT_EPSILON);
EXPECT_NEAR(workplaceAdults[exateppabm::population::Workplace::WORKPLACE_ADULT], target_adult_workplace[exateppabm::population::Workplace::WORKPLACE_ADULT], TARGET_ADULT_EPSILON);
EXPECT_NEAR(workplaceAdults[exateppabm::population::Workplace::WORKPLACE_70_79], target_adult_workplace[exateppabm::population::Workplace::WORKPLACE_70_79], TARGET_ADULT_EPSILON);
EXPECT_NEAR(workplaceAdults[exateppabm::population::Workplace::WORKPLACE_80_PLUS], target_adult_workplace[exateppabm::population::Workplace::WORKPLACE_80_PLUS], TARGET_ADULT_EPSILON);

// @todo - expand test with more edge case coverage
}
