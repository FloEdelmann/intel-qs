find_path(PAPI_PREFIX
    NAMES include/papi.h
)

find_library(PAPI_LIBRARIES
    # Pick the static library first for easier run-time linking.
    NAMES libpapi.so libpapi.a papi
    HINTS ${PAPI_PREFIX}/lib ${HILTIDEPS}/lib
)

find_path(PAPI_INCLUDE_DIRS
    NAMES papi.h
    HINTS ${PAPI_PREFIX}/include ${HILTIDEPS}/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PAPI DEFAULT_MSG
    PAPI_LIBRARIES
    PAPI_INCLUDE_DIRS
)

mark_as_advanced(
    PAPI_PREFIX_DIRS
    PAPI_LIBRARIES
    PAPI_INCLUDE_DIRS
)

