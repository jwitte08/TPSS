message("                                                            ")
message("                            TPSS                            ")
message("            Fast Tensor Product Schwarz Smoothers           ")
message("                                                            ")

cmake_minimum_required(VERSION 2.12.0)

project(TPSS
  LANGUAGES CXX
  DESCRIPTION "Fast Tensor Product Schwarz Smoothers"
  )

################################################################################

if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  # extract the name of the current git branch
  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  # extract the abbr. hash of the last commit
  execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(GIT_BRANCH "")
  set(GIT_COMMIT_HASH "")
endif(EXISTS "${CMAKE_SOURCE_DIR}/.git")

message(STATUS "Current git branch:      ${GIT_BRANCH}")
message(STATUS "Current git commit hash: ${GIT_COMMIT_HASH}")

configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/git_version.h.in
  ${CMAKE_BINARY_DIR}/include/git_version.h
  )

################################################################################

# fill in (python) scripts with CMake information
configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/ct_parameter.py.in
  ${CMAKE_BINARY_DIR}/scripts/ct_parameter.py
  )
configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/generate_exec.py.in
  ${CMAKE_BINARY_DIR}/scripts/generate_exec.py
  )
configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/env.sh.in
  ${CMAKE_BINARY_DIR}/env.sh
  @ONLY
  )

################################################################################

find_package (PythonInterp 3 REQUIRED)

# generates the default compile time parameters
execute_process (
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_BINARY_DIR}/scripts/ct_parameter.py -O ${CMAKE_BINARY_DIR}/include/ct_parameter.h
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

################################################################################

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified.")
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

find_package(deal.II 9.1 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
  )
if(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()
if(NOT DEAL_II_WITH_P4EST)
  message(FATAL_ERROR
    "\nWe require a deal.II installation built with support for p4est but it appears to be missing!\n"
    )
endif()
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# currently TPSS is header-only, that is an INTERFACE library in cmake
add_library(tpss INTERFACE)
target_compile_features(tpss INTERFACE cxx_std_14)

# set the include directory to be build from (BUILD_INTERFACE)
# set the install path the includes are "copied" to (INSTALL_INTERFACE)
# the suffix "_INTERFACE" is only required by interface libraries
target_include_directories(tpss INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/TPSS>
  )

#ADD_SUBDIRECTORY(TPSS) # old

ADD_SUBDIRECTORY(apps)

ADD_SUBDIRECTORY(tests)

