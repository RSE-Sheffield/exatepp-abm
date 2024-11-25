#pragma once

#include <cstdint>
#include <string>
#include <numeric>

namespace exateppabm {
namespace util {

/**
 * Get the device name for the specified device ordinal
 *
 * @param device device ordinal
 * @return string device name
 */

std::string getGPUName(int ordinal);

/**
 * Get the number of streaming multiprocessors for specified device ordinal
 *
 * @param device device ordinal
 * @return device streaming multiprocessor count
 */

std::uint32_t getGPUMultiProcessorCount(int ordinal);

/**
 * Get the total memory for specified device ordinal
 *
 * @param device device ordinal
 * @return device memory in bytes
 */

std::size_t getGPUMemory(int ordinal);

/**
 * Initialise the CUDA context on the specified device to move the CUDA context creation time
 *
 * @param ordinal device ordinal
 */
void initialiseCUDAContext(int ordinal);

/**
 * Get a bool indicating if FLAMEGPU_SEATBELTS were enabled or not.
 *
 * @return bool indicating if FLAMEGPU_SEATBELTS was enabled
 */
bool getSeatbeltsEnabled();

/**
 * Get a string representing the build configuration
 *
 * @return string representation of the build configuration
 */
std::string getCMakeBuildType();


/**
 * Perform an inclusive scan on an interable (i.e. std::array<float>), without using std::inclusive_scan which although part of c++17 is not available in some older stdlib implementations (i.e. gcc 8). 
 */

/**
 * In-place inclusive scan, for libstdc++ which does not support c++17 (i.e. GCC 8)
 * 
 * equivalent to std::inclusive_scan(container.begin(), container.end(), container.begin());
 * 
 * @param container iterable container / something with size() and operator[]
 */
template <typename T>
void naive_inplace_inclusive_scan(T& container) {
    if (container.size() <= 1) {
        return;
    }
    // Naive in-place inclusive scan for libstc++8
    for (size_t i = 1; i < container.size(); ++i) {
        container[i] = container[i - 1] + container[i];
    }
}

/**
 * In-place inclusive scan, using std::inclusive_scan if possible, else a naive implementation.
 * 
 * equivalent to std::inclusive_scan(container.begin(), container.end(), container.begin());
 */
template <typename T>
void inplace_inclusive_scan(T& container) {
#if defined(EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN) && EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN
    std::inclusive_scan(container.begin(), container.end(), container.begin());
#else
    naive_inplace_inclusive_scan(container);
#endif  // defined(EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN) && EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN
}

}  // namespace util
}  // namespace exateppabm
