## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

set(
  _h
  PUJ_ML/Model/Base.h
  PUJ_ML/Model/Linear.h
  PUJ_ML/Model/Logistic.h
  PUJ_ML/Model/Perceptron.h
  )
set(
  _cxx
  PUJ_ML/Model/Base.cxx
  PUJ_ML/Model/Linear.cxx
  PUJ_ML/Model/Logistic.cxx
  PUJ_ML/Model/Perceptron.cxx
  )

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
add_library(PUJ_ML SHARED ${_cxx} ${_h})
generate_export_header(
  PUJ_ML
  EXPORT_MACRO_NAME PUJ_ML_EXPORT
  EXPORT_FILE_NAME PUJ_ML/Export.h
  )
target_link_libraries(PUJ_ML Eigen3::Eigen)

## eof - $RCSfile$
