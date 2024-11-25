#pragma once

#include <cstdint>
#include <string>
#include <numeric>
#include <algorithm>
#include <iterator>

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
 * Inclusive scan from [first, last) writing into d_first (and onwards), for libstdc++ which does not support c++17 (i.e. GCC 8)
 * 
 * equivalent to std::inclusive_scan(first, last, d_first);
 * 
 * @note - this is very naive, and will happily do out of bounds reads/writes. Using GCC >= 8 is greatly preferred
 * 
 * @param first - input iterator for the first element
 * @param last - input iterator for the last element
 * @param d_first destination iterator for the last element
 */
template <typename InputIt, typename OutputIt>
void naive_inclusive_scan(InputIt first, InputIt last, OutputIt d_first) {
    using T = typename std::iterator_traits<InputIt>::value_type;
    T acc = T(); // default initialise T (the inferred type)
    for (; first != last; ++first, ++d_first) {
        acc += *first;
        *d_first = acc;
    }
}

/**
 * Inclusive scan from [first, last) writing into d_first (and onwards), using std::inclusive_scan if possible, else a naive implementation.
 * 
 * equivalent to std::inclusive_scan(first, last, d_first);
 *
 * @param first - input iterator for the first element
 * @param last - input iterator for the last element
 * @param d_first destination iterator for the last element
 */
template <typename InputIt, typename OutputIt>
void inclusive_scan(InputIt first, InputIt last, OutputIt d_first) {
#if defined(EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN) && EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN
    std::inclusive_scan(first, last, d_first);
#else
    naive_inclusive_scan(first, last, d_first);
#endif  // defined(EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN) && EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN
}

/**
 * reduce elements of a container, i.e. std::reduce, for libstdc++ which does not support c++17 (i.e. GCC 8)
 * 
 * equivalent to std::reduce(first, last, init);
 * 
 * @param first - input iterator for the first element
 * @param last - input iterator for the last element
 * @param init - initial value to reduce into (and inferred type)
 * @return reduction (sum) if values between in [first, last)
 */
template <typename InputIt, typename T>
T naive_reduce(InputIt first, InputIt last, T init) {
    for (; first != last; ++first) {
        init += *first;
    }
    return init;
}

/**
 * reduce elements of a container, using std::reduce if possible, else a naive implementation.
 * 
 * equivalent to std::reduce(first, last, init);
 */
template <typename InputIt, typename T>
T reduce(InputIt first, InputIt last, T init) {
#if defined(EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN) && EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN
    return std::reduce(first, last, init);
#else
    return naive_reduce(first, last, init);
#endif  // defined(EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN) && EXATEPP_ABM_USE_STD_INCLUSIVE_SCAN
}

}  // namespace util
}  // namespace exateppabm
