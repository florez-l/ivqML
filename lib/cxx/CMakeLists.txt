## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/lib/cxx
  ${PROJECT_BINARY_DIR}/lib/cxx
  )
add_library(
  PUJ_ML
  SHARED
  PUJ/Model.h
  PUJ/Model/Linear.cxx
  PUJ/Model/Logistic.cxx
  PUJ/Regression.h
  PUJ/Regression/Base.cxx
  PUJ/Regression/MaximumLikelihood.cxx
  PUJ/Regression/MSE.cxx
  )
generate_export_header(
  PUJ_ML
  EXPORT_MACRO_NAME PUJ_ML_EXPORT
  EXPORT_FILE_NAME PUJ_ML_export.h
  )
target_link_libraries(PUJ_ML ${Eigen3_LIBRARIES})

## eof - $RCSfile$
