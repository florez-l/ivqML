## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_BINARY_DIR}/lib/cxx
  ${PROJECT_SOURCE_DIR}/lib/cxx
  )

add_executable(example_LinearModel_00 LinearModel_00.cxx)
target_link_libraries(example_LinearModel_00 PUJ_ML)

add_executable(example_LogisticModel_00 LogisticModel_00.cxx)
target_link_libraries(example_LogisticModel_00 PUJ_ML)

add_executable(example_PerceptronModel_00 PerceptronModel_00.cxx)
target_link_libraries(example_PerceptronModel_00 PUJ_ML)

## eof - $RCSfile$
