## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

file(GLOB_RECURSE _c ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
file(GLOB_RECURSE _cxx ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx)
file(GLOB_RECURSE _h ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE _hxx ${CMAKE_CURRENT_SOURCE_DIR}/*.hxx)
set(_src "${_c};${_cxx};${_h};${_hxx}")
set(_libs atomic ivq::ivq)

include_directories(
  ${PROJECT_BINARY_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib
  )
add_library(ivqML ${ivq_BUILD_TYPE} ${_src})
generate_export_header(
  ivqML
  EXPORT_MACRO_NAME ivqML_EXPORT
  EXPORT_FILE_NAME Export.h
  )
target_link_libraries(ivqML "${_libs}")

## -- Installation rules
install(
  TARGETS ivqML
  EXPORT "${targets_export_name}"
  LIBRARY DESTINATION "lib"
  ARCHIVE DESTINATION "lib"
  RUNTIME DESTINATION "bin"
  INCLUDES DESTINATION "${include_install_dir}"
  )

set(_hdr ${_src})
list(FILTER _hdr INCLUDE REGEX ".*\\.h.*$")
foreach(_h ${_hdr})
  get_filename_component(_d ${_h} DIRECTORY)
  string(REGEX REPLACE "^${CMAKE_CURRENT_SOURCE_DIR}" "" _d ${_d})
  ## string(REGEX REPLACE "^${CMAKE_CURRENT_BINARY_DIR}" "" _f ${_e})
  install(FILES ${_h} DESTINATION include/ivqML${_d})
endforeach(_h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Export.h DESTINATION include/ivqML)

## eof - $RCSfile$
