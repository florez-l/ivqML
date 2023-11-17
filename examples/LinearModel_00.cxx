// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <ivqML/Model/Linear.h>

using _R = long double;
using _L = ivqML::Model::Linear< _R >;

int main( int argc, char** argv )
{
  unsigned int n = 4;
  unsigned int m = 10;

  // A model
  _L model( n );
  model.random_fill( );
  std::cout << "Model: " << model << std::endl;

  // Some random input data
  _L::TMatrix X( m, n );
  X.setRandom( );
  std::cout << "-------------- INPUTS --------------" << std::endl;
  std::cout << X << std::endl;

  std::cout << "-------------- OUTPUTS --------------" << std::endl;
  _L::TMatrix Y = model.evaluate( X );
  std::cout << Y << std::endl;

  return( EXIT_SUCCESS );
}


// eof - $RCSfile$
