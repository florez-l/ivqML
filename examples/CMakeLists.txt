## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_BINARY_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib
  )

add_executable(ivqML_example_Sandbox Sandbox.cxx)
target_link_libraries(ivqML_example_Sandbox ivqML)
vtk_module_autoinit(
  TARGETS ivqML_example_Sandbox
  MODULES ${VTK_LIBRARIES}
  )

set(
  _examples
  AnalyzeImageTexture
  KMeansImageFilter
  PCAImageFilter

  # FeedForwardNetwork_00
  # FeedForwardNetwork_01
  # LinearModel_00
  # LinearModel_AnalyticalFit_CSV
  # LinearModel_ADAMFit_00
  # LinearModel_GradientDescentFit_00
  # LogisticModel_00

  #  ExtractSamplesFrom2DImage
  #  
  #  FeedForwardNetwork_01
  #  FeedForwardNetwork_ADAMFit_Image
  #  FeedForwardNetwork_GradientDescentFit_Image
  #  
  #  LinearModel_AnalyticalFit_CSV
  #  LinearModel_ADAMFit_00
  #  LinearModel_ADAMFit_CSV
  #  LinearModel_GradientDescentFit_00
  #  LinearModel_GradientDescentFit_CSV
  #  
  #  LogisticModel_ADAMFit_CSV
  #  LogisticModel_GradientDescentFit_CSV
  #  MNIST_FeedForwardNetwork_Train
  #  
  )
set(
  _examples_with_vtk
  2DKMeansFromCSV
  )

foreach(_e ${_examples})
  add_executable(ivqML_example_${_e} ${_e}.cxx)
  target_link_libraries(ivqML_example_${_e} ivqML)
endforeach()

foreach(_e ${_examples_with_vtk})
  add_executable(ivqML_example_${_e} ${_e}.cxx)
  target_link_libraries(ivqML_example_${_e} ivqML)
  vtk_module_autoinit(TARGETS ivqML_example_${_e} MODULES ${VTK_LIBRARIES})
endforeach()


## eof - $RCSfile$
