## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_SOURCE_DIR}/lib/cxx
  ${PROJECT_BINARY_DIR}/lib/cxx
  )

add_executable(LinearModel_01 LinearModel_01.cxx)
target_link_libraries(LinearModel_01 Eigen3::Eigen)

add_executable(LinearModel_Fit_01 LinearModel_Fit_01.cxx)
target_link_libraries(LinearModel_Fit_01 Eigen3::Eigen)

add_executable(LinearModel_Fit_02 LinearModel_Fit_02.cxx)
target_link_libraries(LinearModel_Fit_02 Eigen3::Eigen)

## eof - $RCSfile$
