# Minimum CMake version 3.18 for CUDA --std=c++17 
cmake_minimum_required(VERSION 3.18...3.25 FATAL_ERROR)

# This CMakeLists should not be built on its own.
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  message(FATAL_ERROR "${CMAKE_CURRENT_SOURCE_DIR} cannot be the top CMakeLists.txt. Please use the parent directory")
endif()

# We already have a project here, no need to reset one, but set variables for the target names
set(LIBRARY_NAME "${PROJECT_NAME}_lib")
set(BINARY_NAME "${PROJECT_NAME}")

# --------------------------------------------
# Define the static library target
# --------------------------------------------
SET(LIBRARY_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/constants.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/cli.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/disease.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/disease/SEIR.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/exatepp_abm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/input.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/output.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/output/OutputFile.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/output/PerformanceFile.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/output/TimeSeriesFile.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/person.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/population.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/typedefs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/util.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/visualisation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/cli.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/disease/SEIR.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/exatepp_abm.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/input.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/output.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/output/PerformanceFile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/output/TimeSeriesFile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/person.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/population.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/util.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/exateppabm/visualisation.cu
)

flamegpu_add_library("${LIBRARY_NAME}" "${LIBRARY_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" FALSE)

# Find and fetch 3rd party dependencies
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/CLI11.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/fmt.cmake)

# Publicly link against the dependencies, in case they are used in tests. This could probably be private.
target_link_libraries("${LIBRARY_NAME}" PUBLIC CLI11::CLI11)
target_link_libraries("${LIBRARY_NAME}" PUBLIC fmt::fmt-header-only)

# Ensure C++17 is being used (it should be already via flamegpu_add_library)
set_property(TARGET "${LIBRARY_NAME}" PROPERTY CXX_STANDARD 17)

# Windows specific fixes
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Prevent windows.h definition of MIN and MAX macros on windows
    target_compile_definitions(${LIBRARY_NAME} PRIVATE NOMINMAX)
    # Allow use of M_PI anywhere
    target_compile_definitions(${LIBRARY_NAME} PRIVATE _USE_MATH_DEFINES)
endif()

# Suppress warnings from fmt in cuda (unless CMake is new enough for add_subdirectory(... SYSTEM)
if(CMAKE_VERSION VERSION_LESS "3.25.0")
    target_compile_options(${LIBRARY_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=128>")
endif()

# Enable host device constexpr?
target_compile_options(${LIBRARY_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

# Forward on the cmake build type as a macro definition, for performance output files. There's prolly a cleaner way to do this.
target_compile_definitions(${LIBRARY_NAME} PRIVATE "$<$<CONFIG:Release>:CMAKE_BUILD_TYPE=\"Release\">")
target_compile_definitions(${LIBRARY_NAME} PRIVATE "$<$<CONFIG:RelWithDebInfo>:CMAKE_BUILD_TYPE=\"RelWithDebInfo\">")
target_compile_definitions(${LIBRARY_NAME} PRIVATE "$<$<CONFIG:MinSizeRel>:CMAKE_BUILD_TYPE=\"MinSizeRel\">")
target_compile_definitions(${LIBRARY_NAME} PRIVATE "$<$<CONFIG:Debug>:CMAKE_BUILD_TYPE=\"Debug\">")

# Add src directory to include path, publicly so that the target library inherits this dependency (for now).
target_include_directories("${LIBRARY_NAME}" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# --------------------------------------------
# Define the binary target
# --------------------------------------------
set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/main.cu)
flamegpu_add_executable("${BINARY_NAME}" "${SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" FALSE)

# Link against the static library target.
target_link_libraries("${BINARY_NAME}" PRIVATE "${LIBRARY_NAME}")

# --------------------------------------------
# Define shorthand lint target
# --------------------------------------------

# Create a target named lint which calls the generated lint_exatepp_abm and lint_exatepp_abm_lib targets
# Create an alias target `lint` for `lint_fujitsu_iow` out of laziness
if(TARGET lint_${LIBRARY_NAME})
    if(NOT TARGET lint)
        add_custom_target(lint)
        set_target_properties(lint PROPERTIES EXCLUDE_FROM_ALL TRUE)
        flamegpu_set_target_folder(lint "Lint")
    endif()
    add_dependencies(lint lint_${LIBRARY_NAME})
endif()

# Make linting the binary also lint the static library, to keep prior behaviour
if(TARGET lint_${BINARY_NAME} AND TARGET lint_${LIBRARY_NAME})
    add_dependencies(lint_${BINARY_NAME} lint_${LIBRARY_NAME})
endif()