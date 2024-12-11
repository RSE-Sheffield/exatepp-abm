#include <array>
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "fmt/core.h"
#include "exateppabm/util.h"


/**
 * Check that getGPUName doesn't cause an error, and should not return the "unknown" string (although might, but in that case something is wrong with the system)
 */
TEST(TestUtil, getGPUName) {
    // Method returns a string, shouldn't be unknown, as we assume there is atleast one gpu.
    std::string gpuName = exateppabm::util::getGPUName(0);
    ASSERT_NE(gpuName, "");
    ASSERT_NE(gpuName, "unknown");
    // No machines should include INT_MAX GPUs, so it should return "unknown"
    std::string invalid = exateppabm::util::getGPUName(INT_MAX);
    ASSERT_EQ(invalid, "unknown");
}

/**
 * Test the getting of SM count method, as we cannot hardcode the expected number, we can just assume it should be non zero for device 0
 * This is a bit of a flakey test
 */
TEST(TestUtil, getGPUMultiProcessorCount) {
    // Assuming the current machine has a gpu, should return a value >= 0
    auto result = exateppabm::util::getGPUMultiProcessorCount(0);
    ASSERT_GE(result, 0);
    // No machines should include INT_MAX GPUs, so it should return 0
    auto invalid = exateppabm::util::getGPUMultiProcessorCount(INT_MAX);
    ASSERT_EQ(invalid, 0);
}

/**
 * Test that getting GPU memory capacity behaves roughly as expected, for sensible and not-sensible device ordinals.
 */
TEST(TestUtil, getGPUMemory) {
    // Assuming the current machine has a gpu, should return a value >= 0
    auto result = exateppabm::util::getGPUMemory(0);
    ASSERT_GE(result, 0u);
    // No machines should include INT_MAX GPUs, so it should return "unknown"
    auto invalid = exateppabm::util::getGPUMemory(INT_MAX);
    ASSERT_EQ(invalid, 0u);
}

/**
 * Initialising the CUDA context has no direct effects, only side effects which are difficult to observe without digging into the CUDA Driver API.
* For now, just make sure the method doesn't trigger any execeptions
 */
TEST(TestUtil, initialiseCUDAContext) {
    ASSERT_NO_THROW(exateppabm::util::initialiseCUDAContext(0));
}

/**
 * Test that getting the seatbelts value behaves correctly
 */
TEST(TestUtil, getSeatbeltsEnabled) {
    auto value = exateppabm::util::getSeatbeltsEnabled();
    #if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        ASSERT_EQ(value, true);
    #else  // !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        ASSERT_EQ(value, false);
    #endif  // !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
}

/**
 * Test that getting the CMake build type value behaves correctly
 */
TEST(TestUtil, getCMakeBuildType) {
    auto value = exateppabm::util::getCMakeBuildType();
#if defined(CMAKE_BUILD_TYPE)
    ASSERT_EQ(value, CMAKE_BUILD_TYPE);
#else  // defined(CMAKE_BUILD_TYPE)
    // If this occurs, the CMake is broken so we will trigger a failure.
    ASSERT_FALSE(true);
#endif  // defined(CMAKE_BUILD_TYPE)
}


/**
 * Test the naive inplace inclusive_scan implementation for an array of integers
 */
TEST(TestUtil, naive_inplace_inclusive_scan_array_int) {
    constexpr std::uint32_t ELEMENTS = 4;
    std::array<int, ELEMENTS> inout = {{1, 2, 3, 4}};
    const std::array<int, ELEMENTS> expected = {{1, 3, 6, 10}};

    exateppabm::util::naive_inclusive_scan(inout.begin(), inout.end(), inout.begin());

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

    exateppabm::util::naive_inclusive_scan(inout.begin(), inout.end(), inout.begin());

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

    exateppabm::util::naive_inclusive_scan(inout.begin(), inout.end(), inout.begin());

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

    exateppabm::util::inclusive_scan(inout.begin(), inout.end(), inout.begin());

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

    exateppabm::util::inclusive_scan(inout.begin(), inout.end(), inout.begin());

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

    exateppabm::util::inclusive_scan(inout.begin(), inout.end(), inout.begin());

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


/**
 * Test the naive plus reduction an array of integers
 */
TEST(TestUtil, naive_reduce_array_int) {
    constexpr std::uint32_t ELEMENTS = 4;
    std::array<int, ELEMENTS> in = {{1, 2, 3, 4}};
    const int expected = 10;
    auto value = exateppabm::util::naive_reduce(in.begin(), in.end(), 0);
    EXPECT_EQ(value, expected);
}

/**
 * Test the naive plus reduction a vector of unsigned integers
 */
TEST(TestUtil, naive_reduce_vec_uint) {
    std::vector<std::uint32_t> in = {{1, 2, 3, 4}};
    const std::uint32_t expected = 10u;
    auto value = exateppabm::util::naive_reduce(in.begin(), in.end(), 0u);
    EXPECT_EQ(value, expected);
}

/**
 * Test the naive plus reduction a vector of floats, using an arbitrary epsilon value which is good enough for the input values
 */
TEST(TestUtil, naive_reduce_vec_double) {
    std::vector<double> in = {{1.1, 2.2, 3.3, 4.4}};
    const double expected = 11.0;
    auto value = exateppabm::util::naive_reduce(in.begin(), in.end(), 0.);
    constexpr double EPSILON = 1.0e-6f;
    EXPECT_NEAR(value, expected, EPSILON);
}

/**
 * Test the naive or std plus reduction an array of integers
 */
TEST(TestUtil, reduce_array_int) {
    constexpr std::uint32_t ELEMENTS = 4;
    std::array<int, ELEMENTS> in = {{1, 2, 3, 4}};
    const int expected = 10;
    auto value = exateppabm::util::reduce(in.begin(), in.end(), 0);
    EXPECT_EQ(value, expected);
}

/**
 * Test the naive or std plus reduction a vector of unsigned integers
 */
TEST(TestUtil, reduce_vec_uint) {
    std::vector<std::uint32_t> in = {{1, 2, 3, 4}};
    const std::uint32_t expected = 10u;
    auto value = exateppabm::util::reduce(in.begin(), in.end(), 0u);
    EXPECT_EQ(value, expected);
}

/**
 * Test the naive or std plus reduction a vector of floats, using an arbitrary epsilon value which is good enough for the input values
 */
TEST(TestUtil, reduce_vec_double) {
    std::vector<double> in = {{1.1, 2.2, 3.3, 4.4}};
    const double expected = 11.0;
    auto value = exateppabm::util::reduce(in.begin(), in.end(), 0.);
    constexpr double EPSILON = 1.0e-6f;
    EXPECT_NEAR(value, expected, EPSILON);
}

/**
 * Test the std plus reduction an array of integers
 This is just to check the test case behaves too
 */
#if defined(EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN) && EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN
TEST(TestUtil, std_reduce_array_int) {
    constexpr std::uint32_t ELEMENTS = 4;
    std::array<int, ELEMENTS> in = {{1, 2, 3, 4}};
    const int expected = 10;
    auto value = std::reduce(in.begin(), in.end(), 0);
    EXPECT_EQ(value, expected);
}
#else
TEST(TestUtil, DISABLED_std_reduce_array_int) {
}
#endif


/**
 * Test the std plus reduction a vector of unsigned integers.
 * This is just to check the test case behaves too
 */
#if defined(EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN) && EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN
TEST(TestUtil, std_reduce_vec_uint) {
    std::vector<std::uint32_t> in = {{1, 2, 3, 4}};
    const std::uint32_t expected = 10u;
    auto value = std::reduce(in.begin(), in.end(), 0u);
    EXPECT_EQ(value, expected);
}
#else
TEST(TestUtil, DISABLED_std_reduce_vec_uint) {
}
#endif

/**
 * Test the std plus reduction a vector of floats, using an arbitrary epsilon value which is good enough for the input values.
 * This is just to check the test case behaves too
 */
#if defined(EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN) && EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN
TEST(TestUtil, std_reduce_vec_double) {
    std::vector<double> in = {{1.1, 2.2, 3.3, 4.4}};
    const double expected = 11.0;
    auto value = std::reduce(in.begin(), in.end(), 0.);
    constexpr double EPSILON = 1.0e-6f;
    EXPECT_NEAR(value, expected, EPSILON);
}
#else
TEST(TestUtil, DISABLED_std_reduce_vec_double) {
}
#endif
