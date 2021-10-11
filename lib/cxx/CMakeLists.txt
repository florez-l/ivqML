## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_SOURCE_DIR}/lib/cxx
  ${PROJECT_BINARY_DIR}/lib/cxx
  )
add_library(
  PUJ_ML
  SHARED
  PUJ/Traits.h
  PUJ/Model/Linear.cxx
  PUJ/Model/Linear.h
  )
generate_export_header(
  PUJ_ML
  EXPORT_MACRO_NAME PUJ_ML_EXPORT
  EXPORT_FILE_NAME PUJ_ML_export.h
  )
target_link_libraries(PUJ_ML Eigen3::Eigen)

## eof - $RCSfile$
