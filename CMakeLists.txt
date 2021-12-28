## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

## -- Base configuration
cmake_minimum_required(VERSION 3.18)
project(PUJ_ML VERSION 1.0.0.1)

## -- CMake directives
set(CMAKE_CXX_STANDARD 17)
include(GenerateExportHeader)

## -- Find Boost
find_package(Boost REQUIRED program_options)

## -- Find Eigen3
find_package(Eigen3 REQUIRED)

## -- Subdirs
subdirs(lib/cxx)

option(PUJ_ML_BUILD_CXX_EXAMPLES "Build examples?" OFF)
if(PUJ_ML_BUILD_CXX_EXAMPLES)
  subdirs(examples/cxx)
endif(PUJ_ML_BUILD_CXX_EXAMPLES)

## eof - $RCSfile$
