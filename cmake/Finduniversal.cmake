find_path(universal_PREFIX
    NAMES include/universal/number/posit/posit.hpp
)

find_path(universal_INCLUDE_DIRS
    NAMES universal/number/posit/posit.hpp
    HINTS ${universal_PREFIX}/include ${HILTIDEPS}/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(universal DEFAULT_MSG
    universal_INCLUDE_DIRS
)

mark_as_advanced(
    universal_PREFIX_DIRS
    universal_LIBRARIES
    universal_INCLUDE_DIRS
)

