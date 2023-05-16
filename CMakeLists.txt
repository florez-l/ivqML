## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

## -- Base configuration
cmake_minimum_required(VERSION 3.18)
project(ivq VERSION 1.0.0.1)

## -- Set a default build type if none was specified
set(default_build_type "Release")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(default_build_type "Debug")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(
    CMAKE_BUILD_TYPE "${default_build_type}" CACHE
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
find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

## -- Find Boost
find_package(Boost COMPONENTS iostreams system filesystem REQUIRED)

## -- Find gnuplot-iostream
find_file(GNUplot_iostream gnuplot-iostream.h)
get_filename_component(GNUplot_iostream_dir ${GNUplot_iostream} DIRECTORY)
include_directories(${GNUplot_iostream_dir})

## -- Subdirs
subdirs(lib/cxx examples/cxx)

## eof - $RCSfile$
