## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_BINARY_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib
  )

add_executable(ivqML_RegressionFitFromCSV RegressionFitFromCSV.cxx)
target_link_libraries(ivqML_RegressionFitFromCSV ivqML)

## eof - $RCSfile$
