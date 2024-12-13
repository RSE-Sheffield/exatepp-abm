# Minimum CMake version 3.18 for CUDA --std=c++17
cmake_minimum_required(VERSION 3.18...3.25 FATAL_ERROR)

# Enable CTest
enable_testing()

# If we have the binary target, check the binary behaves as intended / produces the correct outputs for given inputs.
if(TARGET exatepp_abm)
    # Test running exatepp_abm without any cli succeeds (returns 0 error code)
    add_test(
        NAME integration.defaultArgs
        COMMAND exatepp_abm
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    set_tests_properties(integration.defaultArgs PROPERTIES LABELS "integration")
    set_tests_properties(integration.defaultArgs PROPERTIES WILL_FAIL FALSE)
endif()

## ------

# Fetch GoogleTest and enable CMake integration
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/GoogleTest.cmake)
include(GoogleTest)

# Handle CMAKE_CUDA_ARCHITECTURES and inject code into the tests project() command
flamegpu_init_cuda_architectures(PROJECT tests)
# Name the project and set languages
project(tests CUDA CXX)
# Include common rules.
include(${FLAMEGPU_ROOT}/cmake/common.cmake)

# List test source files
set(TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/exateppabm/test_household.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/exateppabm/test_network.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/exateppabm/test_util.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/exateppabm/test_workplace.cu
)

# Define output location of binary files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/)
# Declare the new binary target and set required options.
flamegpu_add_executable("${PROJECT_NAME}" "${TESTS_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" FALSE)
# Add the current dir to the include path
target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
# Link against GTest
target_link_libraries("${PROJECT_NAME}" PRIVATE GTest::gtest)
# Link against the static library 
target_link_libraries("${PROJECT_NAME}" PRIVATE exatepp_abm_lib)
# Workaround for incremental rebuilds on MSVC, where device link was not being performed.
# https://github.com/FLAMEGPU/FLAMEGPU2/issues/483
if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.1")
    # Provide the absolute path to the lib file, rather than the relative version cmake provides.
    target_link_libraries(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE:exatepp_abm_lib>")
endif()

# Prevent windows.h definition of MIN and MAX macros on windows
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
endif()

# Put Within Tests filter
flamegpu_set_target_folder("${PROJECT_NAME}" "Tests")
# Also set as startup project (if top level project)
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT "${PROJECT_NAME}")
# Set the default (visual studio) debugger configure_file
set_target_properties("${PROJECT_NAME}" PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
VS_DEBUGGER_COMMAND_ARGUMENTS "$<$<CONFIG:Debug>:--gtest_catch_exceptions=0> --gtest_filter=*")

# Add the tests target as to ctest, optionally using the gtest_discover integration.
if(ENABLE_GTEST_DISCOVER)
    # If GTEST_DISCOVER is enabled, add the unit test executable using it. This results in very long test exeuction times due to CUDA context initialisation per test
    gtest_discover_tests(
        "${PROJECT_NAME}"
        WORKING_DIRECTORY ${PROJECT_DIR}
        TEST_PREFIX "${PROJECT_NAME}."
        PROPERTIES LABELS "unit"
    )
else()
    # Otherwise add the whole test suite as a single test. Use ctest -VV to view per-test results in this case.
    add_test(
        NAME ${PROJECT_NAME}
        COMMAND "${PROJECT_NAME}"
        WORKING_DIRECTORY ${PROJECT_DIR}
    )
    set_tests_properties(${PROJECT_NAME} PROPERTIES LABELS "unit")
endif()

# Add to the exatepp_abm generic lint target.
if(TARGET lint_${PROJECT_NAME} AND TARGET lint)
    add_dependencies(lint lint_${PROJECT_NAME})
endif()
