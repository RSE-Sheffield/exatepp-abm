#include <cstdio>
// Include google test
#include "gtest/gtest.h"
// Include flame gpu telemetry header to prevent every test from triggering a telemetry API call.
#include "flamegpu/io/Telemetry.h"

GTEST_API_ int main(int argc, char **argv) {
    // Disable flamegpu telemetry
    flamegpu::io::Telemetry::disable();
    flamegpu::io::Telemetry::suppressNotice();
    // Run the main google test body
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    auto rtn = RUN_ALL_TESTS();
    // Reset all cuda devices for memcheck / profiling purposes.
    cudaError_t status = cudaSuccess;
    int devices = 0;
    status = cudaGetDeviceCount(&devices);
    if (status == cudaSuccess && devices > 0) {
        for (int device = 0; device < devices; ++device) {
            cudaSetDevice(device);
            cudaDeviceReset();
        }
    }
    return rtn;
}
