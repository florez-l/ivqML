// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <fstream>
#include <ivqML/Model/FeedForwardNetwork.h>

using _R = long double;
using _M = ivqML::Model::FeedForwardNetwork< _R >;

int main( int argc, char** argv )
{
  if( argc < 3 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ]
      << " model_description_file samples" << std::endl;
    return( EXIT_FAILURE );
  } // end if
  std::string d = argv[ 1 ];
  unsigned int m = std::atoi( argv[ 2 ] );

  _M model;
  std::ifstream d_str( d.c_str( ) );
  d_str >> model;
  d_str.close( );

  std::cout << model << std::endl;

  // Some random input data
  _M::TMatrix X( m, model.number_of_inputs( ) );
  X.setRandom( );
  std::cout << "-------------- INPUTS --------------" << std::endl;
  std::cout << X << std::endl;

  std::cout << "-------------- OUTPUTS --------------" << std::endl;
  _M::TMatrix Y( m, model.number_of_outputs( ) );
  model( Y, X );
  std::cout << Y << std::endl;
  std::cout << "---------- BACKPROPAGATION ----------" << std::endl;
  model( Y, X, true );

  return( EXIT_SUCCESS );
}


// eof - $RCSfile$