## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

set(
  _exec
  LogisticRegressionFit
  )

include_directories(
  ${PROJECT_BINARY_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib
  )

foreach(_e ${_exec})
  add_executable(ivqML_${_e} ${_e}.cxx)
  target_link_libraries(ivqML_${_e} ${Boost_LIBRARIES} ivqML)
endforeach()

## eof - $RCSfile$
