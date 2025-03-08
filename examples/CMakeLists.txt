## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================
option(ivqML_BUILD_EXAMPLES "" OFF)

if(ivqML_BUILD_EXAMPLES)
  set(
    _exec
    FeedForwardNeuralNetwork
    )

  include_directories(
    ${PROJECT_BINARY_DIR}/lib
    ${PROJECT_SOURCE_DIR}/lib
    )

  foreach(_e ${_exec})
    add_executable(ivqML_example_${_e} ${_e}.cxx)
    target_link_libraries(ivqML_example_${_e} ${Boost_LIBRARIES} ivqML)
  endforeach()
endif()

## eof - $RCSfile$
