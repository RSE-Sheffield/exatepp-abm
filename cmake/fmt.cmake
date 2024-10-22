include(FetchContent)

# Fetch fmtlib for modern c++ string formatting pre c++20
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG        11.0.2
    GIT_SHALLOW    1
    GIT_PROGRESS   ON
)

# Prevent install rules
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
mark_as_advanced(FORCE FMT_INSTALL)

# Fetch and include FMT, but do not add it to the all target, header-only use is all that is needed.
FetchContent_GetProperties(fmt)
if(NOT fmt_POPULATED)
  FetchContent_Populate(fmt)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.25.0")
    add_subdirectory(${fmt_SOURCE_DIR} ${fmt_BINARY_DIR} EXCLUDE_FROM_ALL SYSTEM)
  else()
    add_subdirectory(${fmt_SOURCE_DIR} ${fmt_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()
