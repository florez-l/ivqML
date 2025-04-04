## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_BINARY_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib
  )

set(
  _execs
  Sandbox
  SimpleFeedForwardSigmoidOutputTest
  SimpleMNISTTest
  )

foreach(_e ${_execs})
  add_executable(ivqML_examples_${_e} ${_e}.cxx)
  target_link_libraries(ivqML_examples_${_e} ivqML)
endforeach()

## eof - $RCSfile$
