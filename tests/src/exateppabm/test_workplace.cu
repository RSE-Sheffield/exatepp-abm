#include <array>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "fmt/core.h"
#include "exateppabm/input.h"
#include "exateppabm/workplace.h"

/**
 * Test getAdultWorkplaceCumulativeProbabilityArray
 */

TEST(TestWorkplace, getAdultWorkplaceCumulativeProbabilityArray) {
const double EPSILON = 1.0e-7;
double child_network_adults = 0;
double elderly_network_adults = 0;
std::array<std::uint64_t, exateppabm::demographics::AGE_COUNT> n_per_age = {};
std::array<double, exateppabm::workplace::WORKPLACE_COUNT> p = {};
std::array<double, exateppabm::workplace::WORKPLACE_COUNT> expected = {};


// With 0 adults in the child or elderly networks, should all be assigned to the adult network, regardless of population counts
child_network_adults = 0;
elderly_network_adults = 0;
n_per_age = {{10, 10, 10, 10, 10, 10, 10, 10, 10}};
expected = {{0.0, 0.0, 1.0, 1.0, 1.0}};
p = exateppabm::workplace::getAdultWorkplaceCumulativeProbabilityArray(child_network_adults, elderly_network_adults, n_per_age);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_SCHOOL_0_9], expected[exateppabm::workplace::WORKPLACE_SCHOOL_0_9], EPSILON);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_SCHOOL_10_19], expected[exateppabm::workplace::WORKPLACE_SCHOOL_10_19], EPSILON);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_ADULT], expected[exateppabm::workplace::WORKPLACE_ADULT], EPSILON);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_70_79], expected[exateppabm::workplace::WORKPLACE_70_79], EPSILON);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_80_PLUS], expected[exateppabm::workplace::WORKPLACE_80_PLUS], EPSILON);

// With 1 adults per member of work networks, and a population which will support this, generate probabilities that would lead to 20% of adults being assigned to each network
child_network_adults = 1.0;
elderly_network_adults = 1.0;
// I.e. 20 children/elderly per network and 100 adults total, 20 in each network in the end
n_per_age = {{20, 20, 20, 20, 20, 20, 20, 20, 20}};
expected = {{0.2, 0.4, 0.6, 0.8, 1.0}};
p = exateppabm::workplace::getAdultWorkplaceCumulativeProbabilityArray(child_network_adults, elderly_network_adults, n_per_age);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_SCHOOL_0_9], expected[exateppabm::workplace::WORKPLACE_SCHOOL_0_9], EPSILON);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_SCHOOL_10_19], expected[exateppabm::workplace::WORKPLACE_SCHOOL_10_19], EPSILON);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_ADULT], expected[exateppabm::workplace::WORKPLACE_ADULT], EPSILON);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_70_79], expected[exateppabm::workplace::WORKPLACE_70_79], EPSILON);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_80_PLUS], expected[exateppabm::workplace::WORKPLACE_80_PLUS], EPSILON);

// With 10% in child networks, and 20% in elderly networks, and 100 adults total, generate the expected (unbalanced) values
child_network_adults = 0.1;
elderly_network_adults = 0.2;
n_per_age = {{20, 20, 0, 0, 20, 0, 0, 20, 10}};
// 2/20 adult, 2/20 adults, 10/20 adults, 4/20 adults, 2/20 adults, but cumulative.
expected = {{0.1, 0.2, 0.7, 0.9, 1.0}};
p = exateppabm::workplace::getAdultWorkplaceCumulativeProbabilityArray(child_network_adults, elderly_network_adults, n_per_age);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_SCHOOL_0_9], expected[exateppabm::workplace::WORKPLACE_SCHOOL_0_9], EPSILON);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_SCHOOL_10_19], expected[exateppabm::workplace::WORKPLACE_SCHOOL_10_19], EPSILON);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_ADULT], expected[exateppabm::workplace::WORKPLACE_ADULT], EPSILON);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_70_79], expected[exateppabm::workplace::WORKPLACE_70_79], EPSILON);
EXPECT_NEAR(p[exateppabm::workplace::WORKPLACE_80_PLUS], expected[exateppabm::workplace::WORKPLACE_80_PLUS], EPSILON);

// @todo - expand test with more edge case coverage
}

/**
 * Test generateWorkplaceForIndividual
 */
TEST(TestWorkplace, generateWorkplaceForIndividual) {
// number of individuals to generate, to do a large enough sample to make sure returned values are in-range.
constexpr std::uint64_t n_samples = 1000u;
constexpr std::uint64_t n_samples_adult = 10000u;
// rng state
std::mt19937_64 rng(0u);
// probability array
std::array<double, exateppabm::workplace::WORKPLACE_COUNT> p_adult_workplace = {{0.1, 0.2, 0.8, 0.9, 1.0}};
// Expected/target counts for adults (non-cumulative)
std::array<double, exateppabm::workplace::WORKPLACE_COUNT> target_adult_workplace = {{
    0.1 * n_samples_adult,
    0.1 * n_samples_adult,
    0.6 * n_samples_adult,
    0.1 * n_samples_adult,
    0.1 * n_samples_adult}};
double TARGET_ADULT_EPSILON = 0.01 * n_samples_adult;

// Repeatedly generate individuals of each age band from a given probability distribution, checking they are always the expected values

// 0-9 always in their workplace
for (std::uint64_t idx = 0; idx < n_samples; ++idx) {
    auto w = exateppabm::workplace::generateWorkplaceForIndividual(exateppabm::demographics::AGE_0_9, p_adult_workplace, rng);
    EXPECT_EQ(w, exateppabm::workplace::Workplace::WORKPLACE_SCHOOL_0_9);
}
// 10-19 always in their workplace
for (std::uint64_t idx = 0; idx < n_samples; ++idx) {
    auto w = exateppabm::workplace::generateWorkplaceForIndividual(exateppabm::demographics::AGE_10_19, p_adult_workplace, rng);
    EXPECT_EQ(w, exateppabm::workplace::Workplace::WORKPLACE_SCHOOL_10_19);
}
// 70-79 always in their workplace
for (std::uint64_t idx = 0; idx < n_samples; ++idx) {
    auto w = exateppabm::workplace::generateWorkplaceForIndividual(exateppabm::demographics::AGE_70_79, p_adult_workplace, rng);
    EXPECT_EQ(w, exateppabm::workplace::Workplace::WORKPLACE_70_79);
}
// 80+ always in their workplace
for (std::uint64_t idx = 0; idx < n_samples; ++idx) {
    auto w = exateppabm::workplace::generateWorkplaceForIndividual(exateppabm::demographics::AGE_80, p_adult_workplace, rng);
    EXPECT_EQ(w, exateppabm::workplace::Workplace::WORKPLACE_80_PLUS);
}
// Adults are randomly assigned subject to the cumulative probability distribution.
std::array<std::uint64_t, exateppabm::workplace::WORKPLACE_COUNT> workplaceAdults = {0};
for (std::uint64_t idx = 0; idx < n_samples_adult; ++idx) {
    // Just check a single age demo for simlicity
    auto w = exateppabm::workplace::generateWorkplaceForIndividual(exateppabm::demographics::AGE_40_49, p_adult_workplace, rng);
    // Ensure in range
    EXPECT_GE(w, 0);
    EXPECT_LT(w, exateppabm::workplace::WORKPLACE_COUNT);
    // Increment the counter
    ++workplaceAdults[w];
}
// Check that each counter is roughly correct.
EXPECT_NEAR(workplaceAdults[exateppabm::workplace::Workplace::WORKPLACE_SCHOOL_0_9], target_adult_workplace[exateppabm::workplace::Workplace::WORKPLACE_SCHOOL_0_9], TARGET_ADULT_EPSILON);
EXPECT_NEAR(workplaceAdults[exateppabm::workplace::Workplace::WORKPLACE_SCHOOL_10_19], target_adult_workplace[exateppabm::workplace::Workplace::WORKPLACE_SCHOOL_10_19], TARGET_ADULT_EPSILON);
EXPECT_NEAR(workplaceAdults[exateppabm::workplace::Workplace::WORKPLACE_ADULT], target_adult_workplace[exateppabm::workplace::Workplace::WORKPLACE_ADULT], TARGET_ADULT_EPSILON);
EXPECT_NEAR(workplaceAdults[exateppabm::workplace::Workplace::WORKPLACE_70_79], target_adult_workplace[exateppabm::workplace::Workplace::WORKPLACE_70_79], TARGET_ADULT_EPSILON);
EXPECT_NEAR(workplaceAdults[exateppabm::workplace::Workplace::WORKPLACE_80_PLUS], target_adult_workplace[exateppabm::workplace::Workplace::WORKPLACE_80_PLUS], TARGET_ADULT_EPSILON);

// @todo - expand test with more edge case coverage
}
