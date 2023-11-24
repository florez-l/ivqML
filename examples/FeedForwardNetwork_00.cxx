// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <ivqML/Model/FeedForwardNetwork.h>

using _R = long double;
using _M = ivqML::Model::FeedForwardNetwork< _R >;

int main( int argc, char** argv )
{
  unsigned int m = 10;

  _M model;
  model.add_layer( 8, 4, "ReLU" );
  model.add_layer( 2, "ReLU" );
  model.add_layer( 1, "siGmoID" );
  model.init( );

  std::cout << model << std::endl;

  // Some random input data
  _M::TMatrix X( m, model.number_of_inputs( ) );
  X.setRandom( );
  std::cout << "-------------- INPUTS --------------" << std::endl;
  std::cout << X << std::endl;

  std::cout << "-------------- OUTPUTS --------------" << std::endl;
  _M::TMatrix Y = model.evaluate( X );
  std::cout << Y << std::endl;

  return( EXIT_SUCCESS );
}


// eof - $RCSfile$
