################################################################################
# Construction of IQS as a shared library
################################################################################

#message(STATUS "Source dir ${PROJECT_SOURCE_DIR} , while modules are in ${CMAKE_MODULE_PATH}")
#message(STATUS "Binary dir ${CMAKE_BINARY_DIR} , and project_bin ${CMAKE_CURRENT_BINARY_DIR}")

# Download and unpack eigen library at configure time.
cmake_minimum_required(VERSION 3.12.0)
include(../cmake/ExternalProject.cmake)
ExternalProject_Add(deps-eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  PREFIX            "${CMAKE_BINARY_DIR}/deps/eigen-prefix/"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  GIT_SHALLOW 1
)

message(STATUS "Eigen library is downloaded in: ${PROJECT_BINARY_DIR}/deps/eigen-prefix/src/eigen/")

################################################################################

set(IQS_FILES 
  chi_matrix.cpp
  highperfkernels.cpp
  spec_kernels.cpp
  qureg_apply1qubitgate.cpp
  qureg_apply2qubitgate.cpp
  qureg_applyctrl1qubitgate.cpp
  qureg_applydiag.cpp
  qureg_apply_channel.cpp
  qureg_applyswap.cpp
  qureg_applytoffoli.cpp
  qureg_expectval.cpp
  qureg_fusion.cpp
  qureg_init.cpp
  qureg_measure.cpp
  qureg_noisysimul.cpp
  qureg_permute.cpp
  qureg_utils.cpp
  qureg_version.cpp
  mpi_env.cpp
  mpi_exception.cpp
  utils.cpp
  rng_utils.cpp
  gate_spec.cpp
  CACHE INTERNAL "" FORCE)

add_library(iqs SHARED ${IQS_FILES})

add_dependencies(iqs deps-eigen)

################################################################################
# Link Universal library.
################################################################################

set(UNIVERSAL_USE_OMP ON) # enable OpenMP parallelization in Universal library
find_package(UNIVERSAL CONFIG)
if (UNIVERSAL_FOUND)
    message(STATUS "Universal UNIVERSAL_FOUND = '${UNIVERSAL_FOUND}'")
    message(STATUS "Universal UNIVERSAL_INCLUDE_DIRS = '${UNIVERSAL_INCLUDE_DIRS}'")
    message(STATUS "Universal UNIVERSAL_LIBRARY_DIRS = '${UNIVERSAL_LIBRARY_DIRS}'")
    message(STATUS "Universal UNIVERSAL_LIBRARIES = '${UNIVERSAL_LIBRARIES}'")
    message(STATUS "Universal UNIVERSAL_VERSION = '${UNIVERSAL_VERSION}'")
    message(STATUS "Universal UNIVERSAL_MAJOR_VERSION = '${UNIVERSAL_MAJOR_VERSION}'")
    message(STATUS "Universal UNIVERSAL_MINOR_VERSION = '${UNIVERSAL_MINOR_VERSION}'")
    message(STATUS "Universal UNIVERSAL_SUBMINOR_VERSION = '${UNIVERSAL_SUBMINOR_VERSION}'")
    message(STATUS "Universal universal_LIBRARIES = '${universal_LIBRARIES}'")
    message(STATUS "Universal universal_INCLUDE_DIRS = '${universal_INCLUDE_DIRS}'")
    message(STATUS "Universal UNIVERSAL_CXX_DEFINITIONS = '${UNIVERSAL_CXX_DEFINITIONS}'")
    # message(STATUS "Universal LIBS = x${UNIVERSAL_LIBRARIES} FLAGS=${UNIVERSAL_LDFLAGS}")
    # link_libraries("${UNIVERSAL_LIBRARIES}")
    target_include_directories(iqs INTERFACE universal_INCLUDE_DIRS)
    # target_link_libraries(iqs universal_LIBRARIES)
else()
    message(FATAL_ERROR "Unable to locate Universal library.")
endif()

target_include_directories(iqs PUBLIC ../include)
target_include_directories(iqs PUBLIC ${CMAKE_BINARY_DIR}/deps/eigen-prefix/src/deps-eigen/Eigen)
target_include_directories(iqs INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

# When using ICC, there is a verbose warning from the external library 'eigen'. Disable the warning:
if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
    target_compile_options(iqs PUBLIC -wd2196)
endif()

set_target_properties(iqs
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    POSITION_INDEPENDENT_CODE 1
)

#add_subdirectory(interface)
