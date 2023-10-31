## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_BINARY_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib
  )

set(
  _simple_examples
  LinearModel_00
  LinearModel_ADAMFit_00
  LinearModel_GradientDescentFit_00
  LogisticModel_00
  )
foreach(_e ${_simple_examples})
  add_executable(ivqML_example_${_e} ${_e}.cxx)
  target_link_libraries(ivqML_example_${_e} ivqML)
endforeach()

## eof - $RCSfile$
