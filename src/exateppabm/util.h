#pragma once

#include <cstdint>
#include <string>

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

}  // namespace util
}  // namespace exateppabm
