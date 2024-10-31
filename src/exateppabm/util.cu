#include "exateppabm/util.h"

#include <cuda.h>

#include <cstdint>
#include <string>

#include "fmt/core.h"

namespace exateppabm {
namespace util {

std::string getGPUName(int ordinal) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, ordinal) == cudaSuccess) {
        return std::string(prop.name);
    } else {
        return "unknown";
    }
}

std::uint32_t getGPUMultiProcessorCount(int ordinal) {
    int value = 0;
    // don't check the error, if the ordinal is bad it will be caught elsewhere
    cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, ordinal);
    return value;
}

std::size_t getGPUMemory(int ordinal) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, ordinal) == cudaSuccess) {
        return prop.totalGlobalMem;
    } else {
        return 0;
    }
}

void initialiseCUDAContext(int ordinal) {
    int device_count = 0;
    // don't report any errors, they will be found later nicely.
    if (cudaGetDeviceCount(&device_count) == cudaSuccess) {
        if (cudaSetDevice(ordinal) == cudaSuccess) {
            cudaFree(nullptr);
        }
    }
}

bool getSeatbeltsEnabled() {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    return true;
#else  // !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    return false;
#endif  // !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
}

std::string getCMakeBuildType() {
#if defined(CMAKE_BUILD_TYPE)
    return CMAKE_BUILD_TYPE;
#else  // defined(CMAKE_BUILD_TYPE)
    return "";
#endif  // defined(CMAKE_BUILD_TYPE)
}

}  // namespace util
}  // namespace exateppabm
