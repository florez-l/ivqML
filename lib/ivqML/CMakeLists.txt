## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

file(GLOB_RECURSE _c ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
file(GLOB_RECURSE _cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE _cxx ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx)
file(GLOB_RECURSE _h ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE _hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE _hxx ${CMAKE_CURRENT_SOURCE_DIR}/*.hxx)

include_directories(
  ${PROJECT_BINARY_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib
  )

add_library(ivqML SHARED "${_c};${_cpp};${_cxx}")
generate_export_header(
  ivqML
  EXPORT_MACRO_NAME ivqML_EXPORT
  EXPORT_FILE_NAME Export.h
  )
target_link_libraries(ivqML ivq::ivq)

## eof - $RCSfile$
