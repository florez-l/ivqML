## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_SOURCE_DIR}/lib/cxx
  ${PROJECT_BINARY_DIR}/lib/cxx
  )

add_executable(LinearRegressionModel_01 LinearRegressionModel_01.cxx)
target_link_libraries(LinearRegressionModel_01 Eigen3::Eigen)

add_executable(LinearRegressionModel_Fit_01 LinearRegressionModel_Fit_01.cxx)
target_link_libraries(LinearRegressionModel_Fit_01 Eigen3::Eigen)

add_executable(LinearRegressionModel_Fit_02 LinearRegressionModel_Fit_02.cxx)
target_link_libraries(LinearRegressionModel_Fit_02 Eigen3::Eigen)

add_executable(LinearRegressionModel_Fit_03 LinearRegressionModel_Fit_03.cxx)
target_link_libraries(LinearRegressionModel_Fit_03 Eigen3::Eigen ${Boost_LIBRARIES})

add_executable(LogisticRegressionModel_Fit_01 LogisticRegressionModel_Fit_01.cxx)
target_link_libraries(LogisticRegressionModel_Fit_01 Eigen3::Eigen)

add_executable(LogisticRegressionModel_Fit_02 LogisticRegressionModel_Fit_02.cxx)
target_link_libraries(LogisticRegressionModel_Fit_02 Eigen3::Eigen)

subdirs(minesweeper)

## eof - $RCSfile$
