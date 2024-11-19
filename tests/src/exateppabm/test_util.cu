#include <array>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "fmt/core.h"
#include "exateppabm/util.h"

/**
 * Test the naive inplace inclusive_scan implementation for an array of integers
 */
TEST(TestUtil, naive_inplace_inclusive_scan_array_int) {
    constexpr std::uint32_t ELEMENTS = 4;
    std::array<int, ELEMENTS> inout = {{1, 2, 3, 4}};
    const std::array<int, ELEMENTS> expected = {{1, 3, 6, 10}};

    exateppabm::util::naive_inplace_inclusive_scan(inout);

    for (std::uint32_t e = 0; e < inout.size(); e++) {
        EXPECT_EQ(inout[e], expected[e]);
    }
}

/**
 * Test the naive inplace inclusive_scan implementation for a vector of unsigned integers
 */
TEST(TestUtil, naive_inplace_inclusive_scan_vec_uint) {
    std::vector<std::uint32_t> inout = {{1, 2, 3, 4}};
    const std::vector<std::uint32_t> expected = {{1, 3, 6, 10}};

    exateppabm::util::naive_inplace_inclusive_scan(inout);

    for (std::uint32_t e = 0; e < inout.size(); e++) {
        EXPECT_EQ(inout[e], expected[e]);
    }
}

/**
 * Test the naive inplace inclusive_scan implementation for a vector of floats, using an arbitrary epsilon value which is good enough for the input values
 */
TEST(TestUtil, naive_inplace_inclusive_scan_vec_double) {
    std::vector<float> inout = {{1.1, 2.2, 3.3, 4.4}};
    const std::vector<float> expected = {{1.1, 3.3, 6.6, 11.0}};

    exateppabm::util::naive_inplace_inclusive_scan(inout);

    constexpr float EPSILON = 1.0e-6f;

    for (float e = 0; e < inout.size(); e++) {
        EXPECT_NEAR(inout[e], expected[e], EPSILON);
    }
}

/**
 * Test the naive or std inplace inclusive_scan implementation for an array of integers
 */
TEST(TestUtil, inplace_inclusive_scan_array_int) {
    constexpr std::uint32_t ELEMENTS = 4;
    std::array<int, ELEMENTS> inout = {{1, 2, 3, 4}};
    const std::array<int, ELEMENTS> expected = {{1, 3, 6, 10}};

    exateppabm::util::inplace_inclusive_scan(inout);

    for (std::uint32_t e = 0; e < inout.size(); e++) {
        EXPECT_EQ(inout[e], expected[e]);
    }
}

/**
 * Test the naive or std inplace inclusive_scan implementation for a vector of unsigned integers
 */
TEST(TestUtil, inplace_inclusive_scan_vec_uint) {
    std::vector<std::uint32_t> inout = {{1, 2, 3, 4}};
    const std::vector<std::uint32_t> expected = {{1, 3, 6, 10}};

    exateppabm::util::inplace_inclusive_scan(inout);

    for (std::uint32_t e = 0; e < inout.size(); e++) {
        EXPECT_EQ(inout[e], expected[e]);
    }
}

/**
 * Test the naive or std inplace inclusive_scan implementation for a vector of floats, using an arbitrary epsilon value which is good enough for the input values
 */
TEST(TestUtil, inplace_inclusive_scan_vec_double) {
    std::vector<float> inout = {{1.1, 2.2, 3.3, 4.4}};
    const std::vector<float> expected = {{1.1, 3.3, 6.6, 11.0}};

    exateppabm::util::inplace_inclusive_scan(inout);

    constexpr float EPSILON = 1.0e-6f;

    for (float e = 0; e < inout.size(); e++) {
        EXPECT_NEAR(inout[e], expected[e], EPSILON);
    }
}


/**
 * Test the std inplace inclusive_scan implementation for an array of integers
 This is just to check the test case behaves too
 */
#if defined(EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN) && EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN
TEST(TestUtil, std_inclusive_scan_array_int) {
    constexpr std::uint32_t ELEMENTS = 4;
    std::array<int, ELEMENTS> inout = {{1, 2, 3, 4}};
    const std::array<int, ELEMENTS> expected = {{1, 3, 6, 10}};

    std::inclusive_scan(inout.begin(), inout.end(), inout.begin());

    for (std::uint32_t e = 0; e < inout.size(); e++) {
        EXPECT_EQ(inout[e], expected[e]);
    }
}
#else
TEST(TestUtil, DISABLED_std_inclusive_scan_array_int) {
}
#endif


/**
 * Test the std inplace inclusive_scan implementation for a vector of unsigned integers.
 * This is just to check the test case behaves too
 */
#if defined(EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN) && EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN
TEST(TestUtil, std_inclusive_scan_vec_uint) {
    std::vector<std::uint32_t> inout = {{1, 2, 3, 4}};
    const std::vector<std::uint32_t> expected = {{1, 3, 6, 10}};

    std::inclusive_scan(inout.begin(), inout.end(), inout.begin());

    for (std::uint32_t e = 0; e < inout.size(); e++) {
        EXPECT_EQ(inout[e], expected[e]);
    }
}
#else
TEST(TestUtil, DISABLED_std_inclusive_scan_vec_uint) {
}
#endif

/**
 * Test the std inplace inclusive_scan implementation for a vector of floats, using an arbitrary epsilon value which is good enough for the input values.
 * This is just to check the test case behaves too
 */
#if defined(EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN) && EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN
TEST(TestUtil, std_inclusive_scan_vec_double) {
    std::vector<float> inout = {{1.1, 2.2, 3.3, 4.4}};
    const std::vector<float> expected = {{1.1, 3.3, 6.6, 11.0}};

    std::inclusive_scan(inout.begin(), inout.end(), inout.begin());

    constexpr float EPSILON = 1.0e-6f;

    for (float e = 0; e < inout.size(); e++) {
        EXPECT_NEAR(inout[e], expected[e], EPSILON);
    }
}
#else
TEST(TestUtil, DISABLED_std_inclusive_scan_vec_double) {
}
#endif