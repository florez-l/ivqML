## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

file(GLOB_RECURSE _c ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
file(GLOB_RECURSE _cxx ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx)
file(GLOB_RECURSE _h ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE _hxx ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
set(_src "${_c};${_cxx};${_h};${_hxx}")
set(_libs Eigen3::Eigen)

include_directories(
  ${PROJECT_BINARY_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib
  )
add_library(ivqML ${ivqML_BUILD_TYPE} ${_src})
generate_export_header(
  ivqML
  EXPORT_MACRO_NAME ivqML_EXPORT
  EXPORT_FILE_NAME Export.h
  )
target_link_libraries(ivqML "${_libs}")

## eof - $RCSfile$
