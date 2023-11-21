## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_BINARY_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib
  )

set(
  _simple_examples
  FeedForwardNetwork_00
  FeedForwardNetwork_01
  LinearModel_00
  LinearModel_AnalyticalFit_CSV
  LinearModel_ADAMFit_00
  LinearModel_ADAMFit_CSV
  LinearModel_GradientDescentFit_00
  LinearModel_GradientDescentFit_CSV
  LogisticModel_00
  LogisticModel_ADAMFit_00
  LogisticModel_GradientDescentFit_00
  )
foreach(_e ${_simple_examples})
  add_executable(ivqML_example_${_e} ${_e}.cxx)
  target_link_libraries(ivqML_example_${_e} ivqML)
endforeach()

## eof - $RCSfile$
