## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_SOURCE_DIR}/lib/cxx
  ${PROJECT_BINARY_DIR}/lib/cxx
  )

set(
  _examples
  LinearRegressionModel_01
  LinearRegressionModel_Fit_01
  LinearRegressionModel_Fit_02
  LinearRegressionModel_Fit_03
  LogisticRegressionModel_Fit_01
  LogisticRegressionModel_Fit_02
  FeedForwardNeuralNetwork_01
  FeedForwardNeuralNetwork_02
  FeedForwardNeuralNetwork_Fit_02
  FeedForwardNeuralNetwork_Fit_03
  )
foreach(_e ${_examples})
  add_executable(${_e} ${_e}.cxx)
  target_link_libraries(${_e} Eigen3::Eigen ${Boost_LIBRARIES})
endforeach()

subdirs(minesweeper)

## eof - $RCSfile$
