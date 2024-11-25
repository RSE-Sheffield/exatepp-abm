# Minimum CMake version 3.18 for CUDA --std=c++17
cmake_minimum_required(VERSION 3.18...3.25 FATAL_ERROR)

# Optionally set the version of flamegpu which should be used, ideally a tag (i.e. `v2.0.0-rc`) or branch name, or potentially a commit hash.
set(FLAMEGPU_VERSION "v2.0.0-rc.2" CACHE STRING "FLAMEGPU/FLAMEGPU2 git branch or tag to use")
# If the above version is a hash instead, also set FLAMEGPU_VERSION_ALLOW_HASH to ON
# set(FLAMEGPU_VERSION_ALLOW_HASH "ON")

# Manually specify the FLAMEGPU_VISUALISATION option to provide it prior to original configuration and allow the default to be overridden in the downstream project
option(FLAMEGPU_VISUALISATION "Enable FLAMEGPU visualisation support" OFF)

# Our core dependency is FLAMEGPU2 lib, first lets find it
include(${CMAKE_CURRENT_LIST_DIR}/cmake/flamegpu2.cmake)

# Handle CMAKE_CUDA_ARCHITECTURES gracefully, passing the project name for code-injection
include(${FLAMEGPU_ROOT}/cmake/CUDAArchitectures.cmake)
flamegpu_init_cuda_architectures(PROJECT exatepp_abm)

# Name the project and set languages, this must be done after flamegpu_init_cuda_architectures
project(exatepp_abm CUDA CXX)

# Detect if we are the top level CMake file or not (for CMake < 3.21)
if(CMAKE_VERSION VERSION_LESS "3.21" AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(PROJECT_IS_TOP_LEVEL ON)
endif()

# Option to enable building / enabling tests
option(BUILD_TESTING "Build the testing tree." OFF)
# Option to enable google test test discovery
cmake_dependent_option(ENABLE_GTEST_DISCOVER "Enable GTEST_DISCOVER for more detailed ctest output without -VV. This dramatically increases test suite runtime to CUDA context initialisation." OFF "BUILD_TESTING" OFF)

# Include common rules from the FLAMEGPU/FLAMEGPU2 repositories CMake
include(${FLAMEGPU_ROOT}/cmake/common.cmake)

# Define output location of binary files
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/)

# Add the child CMakeLists.txt which defines the static library and binary.
add_subdirectory(src "${PROJECT_BINARY_DIR}/exatepp_abm")

# --- 
# If testing is enabled and we are the top level cmake, add the test directory.
if(BUILD_TESTING AND PROJECT_IS_TOP_LEVEL) # OR FRE_BUILD_TESTING @todo?
    enable_testing()
    add_subdirectory(tests)
endif()