## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

## -- Base configuration
cmake_minimum_required(VERSION 3.18)
project(PUJ_ML VERSION 1.0.0.0)

## -- Output dirs
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## -- CMake directives
include(GenerateExportHeader)

## -- Find Boost
find_package(Boost REQUIRED program_options)

## -- Find Eigen (3.4)
find_package(Eigen3 3.4 REQUIRED)

## -- Source code
subdirs(lib/cxx examples)

## eof - $RCSfile$
