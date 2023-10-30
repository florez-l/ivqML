## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

## -- Base configuration
cmake_minimum_required(VERSION 3.18)
project(ivqML VERSION 1.0.0.1)

## -- Decide whether links are shared or static
if(WIN32 OR APPLE OR MINGW OR MSYS OR CYGWIN)
  set(ivqML_BUILD_TYPE STATIC)
else()
  set(ivqML_BUILD_TYPE SHARED)
endif()

## -- Set a default build type if none was specified
set(def_bld "Release")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(def_bld "Debug")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${def_bld}' as none was specified.")
  set(
    CMAKE_BUILD_TYPE "${def_bld}" CACHE
    STRING "Choose the type of build." FORCE
    )
  # Set the possible values of build type for cmake-gui
  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
    )
endif()

## -- Output dirs
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## -- CMake directives
include(GenerateExportHeader)

## -- Find Eigen
find_package(Eigen3 3.4 REQUIRED)
option(ivqML_OLD_EIGEN "Use an Eigen3 old version?" OFF)
if(ivqML_OLD_EIGEN)
  add_compile_definitions(ivqML_OLD_EIGEN)
endif(ivqML_OLD_EIGEN)

## -- Find OpenMP, for multiprocessing
find_package(OpenMP QUIET)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(
    CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}"
    )
endif()

## -- Subdirs
subdirs(lib examples)

## eof - $RCSfile$
