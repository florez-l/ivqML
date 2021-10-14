## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

set(
  _examples
  LinearModel_00
  LinearModel_01
  LinearModel_02
  LogisticModel_00
  LogisticModel_01
  )

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/lib/cxx
  ${PROJECT_BINARY_DIR}/lib/cxx
  )
foreach(_e ${_examples})
  add_executable(example_${_e} ${_e}.cxx)
  target_link_libraries(example_${_e} PUJ_ML)
endforeach(_e)

## eof - $RCSfile$
