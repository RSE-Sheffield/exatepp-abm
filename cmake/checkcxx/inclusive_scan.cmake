# only run this once
include_guard(GLOBAL)

# CMake function to check if std::inclusive_scan is supported or not by the current CXX compiler
function(check_inclusive_scan)

    # Ensure c++ is enabled and supported
    include(CheckLanguage)
    check_language(CXX)

    # If this has already been checked for the current CMakeCache, don't rerun.
    if(DEFINED EXATEPP_ABM_check_inclusive_scan_RESULT)
        return()
    endif()

    # Try and compile the sample in c++17
    try_compile(
        HAS_STD_INCLUSIVE_SCAN
        "${CMAKE_CURRENT_BINARY_DIR}/try_compile/"
        "${CMAKE_CURRENT_LIST_DIR}/inclusive_scan.cpp"
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED "ON"
    )

    # If notsupported, mark store this in the parent cache
    if (NOT HAS_STD_INCLUSIVE_SCAN) 
        set(EXATEPP_ABM_check_inclusive_scan_RESULT "NO" PARENT_SCOPE)
        return()
    endif()
    set(EXATEPP_ABM_check_inclusive_scan_RESULT "YES" PARENT_SCOPE)
endfunction()

check_inclusive_scan()
