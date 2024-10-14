##############
# GOOGLETEST #
##############

include(FetchContent)
cmake_policy(SET CMP0079 NEW)

# Googltest newer than 389cb68b87193358358ae87cc56d257fd0d80189 (included in release-1.11.0) or newer is required for CMake >= 3.19
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    # Suppress installation target, as this makes a warning
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    mark_as_advanced(FORCE INSTALL_GTEST)
    mark_as_advanced(FORCE BUILD_GMOCK)
    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    # Set the folder to use
    if (DEFINED flamegpu_set_target_folder)
        flamegpu_set_target_folder(gtest "Tests/Dependencies")
    endif()
    # Disable compiler warnings
    if(TARGET gtest)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_compile_options(gtest PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/W0>")
        else()
            target_compile_options(gtest PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-w>")
        endif()
        # Always tell nvcc to disable warnings
        target_compile_options(gtest PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-w>")
    endif()
endif()

