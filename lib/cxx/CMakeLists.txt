## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

set(
  _h
  PUJ/Helpers/CSV.h
  PUJ/Model/Base.h
  PUJ/Model/Linear.h
  PUJ/Model/Logistic.h
  PUJ/Model/Perceptron.h
  )
set(
  _cxx
  PUJ/Helpers/CSV.cxx
  PUJ/Model/Base.cxx
  PUJ/Model/Linear.cxx
  PUJ/Model/Logistic.cxx
  PUJ/Model/Perceptron.cxx
  )

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
add_library(PUJ_ML SHARED ${_cxx} ${_h})
generate_export_header(
  PUJ_ML
  EXPORT_MACRO_NAME PUJ_EXPORT
  EXPORT_FILE_NAME PUJ/Export.h
  )
target_link_libraries(PUJ_ML ${Boost_LIBRARIES} Eigen3::Eigen)

## eof - $RCSfile$
