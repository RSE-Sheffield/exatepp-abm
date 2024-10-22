include(FetchContent)

# Fetch CLI11 for command line interface
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11
    GIT_TAG        v2.4.2
    GIT_SHALLOW    1
    GIT_PROGRESS   ON
)
FetchContent_MakeAvailable(cli11)

# Set CL11 include dir to be a system dir to avoid warnings. Can't use https://cmake.org/cmake/help/latest/prop_tgt/SYSTEM.html till CMake >= 3.25
get_target_property(CLI11_IID CLI11 INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(CLI11 PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${CLI11_IID}")

# Hide some CLI11 options from CMake UI
mark_as_advanced(CLI11_PRECOMPILED)
mark_as_advanced(CLI11_SANITIZERS)
mark_as_advanced(CLI11_SINGLE_FILE)
mark_as_advanced(CLI11_WARNINGS_AS_ERRORS)
