## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_BINARY_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib
  )

add_executable(ivqML_examples_Sandbox Sandbox.cxx)
target_link_libraries(ivqML_examples_Sandbox ivqML)

## eof - $RCSfile$
