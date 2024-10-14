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
    ${CMAKE_CURRENT_SOURCE_DIR}/exatepp_abm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/exatepp_abm.cu
)

flamegpu_add_library("${LIBRARY_NAME}" "${LIBRARY_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" FALSE)


if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Prevent windows.h definition of MIN and MAX macros on windows
    target_compile_definitions(${LIBRARY_NAME} PRIVATE NOMINMAX)
    # Allow use of M_PI anywhere
    target_compile_definitions(${LIBRARY_NAME} PRIVATE _USE_MATH_DEFINES)
endif()

# Add src directory to include path
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