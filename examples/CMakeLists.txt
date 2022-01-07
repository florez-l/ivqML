## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_BINARY_DIR}/lib/cxx
  ${PROJECT_SOURCE_DIR}/lib/cxx
  )
  
set(
  _examples
  LinearModel_00
  LinearModel_FitGradientDescent_00
  LogisticModel_00
  PerceptronModel_00
  )

foreach(_e ${_examples})
  add_executable(example_${_e} ${_e}.cxx)
  target_link_libraries(example_${_e} PUJ_ML)
endforeach(_e)

## eof - $RCSfile$
