## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

## -- Base configuration
cmake_minimum_required(VERSION 3.18)
project(ivqML VERSION 1.0.0.1)

# -- Output dirs
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# -- CMake directives and base applications
include(GenerateExportHeader)

# -- Find required packages
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(ivq REQUIRED)
set(ivqML_OpenMP_LINK "")
if(OpenMP_CXX_FOUND)
  set(ivqML_OpenMP_LINK "OpenMP::OpenMP_CXX")
endif()

## -- Subdirs
subdirs(examples lib/ivqML)
# TODO: include(cmake/ivqMLInstall.cmake)

## eof - $RCSfile$
