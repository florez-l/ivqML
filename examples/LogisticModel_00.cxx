// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <ivqML/Model/Logistic.h>

using _R = long double;
using _L = ivqML::Model::Logistic< _R >;

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
  _L::TMatrix Y( m, 1 );
  model( Y, X );
  std::cout << Y << std::endl;

  std::cout << "-------------- THRESHOLDS --------------" << std::endl;
  _L::TMatrix Z( m, 1 );
  model.threshold( Z, X );
  std::cout << Z << std::endl;

  return( EXIT_SUCCESS );
}


// eof - $RCSfile$
