// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <ivqML/Model/Regression/Logistic.h>

using TReal = long double;
using TModel = ivqML::Model::Regression::Logistic< TReal >;

int main( int argc, char** argv )
{
  unsigned int n = 4;
  unsigned int m = 10;

  // A model
  TModel model( n );
  model.random_fill( );
  std::cout << "Model: " << model << std::endl;

  // Some random input data
  TModel::TMatrix X( n, m );
  X.setRandom( );
  std::cout << "-------------- INPUTS --------------" << std::endl;
  std::cout << X << std::endl;

  std::cout << "-------------- OUTPUTS --------------" << std::endl;
  TModel::TMatrix Y = model.evaluate( X );
  std::cout << Y << std::endl;

  std::cout << "-------------- THRESHOLDS --------------" << std::endl;
  TModel::TMatrix Z = model.threshold( X );
  std::cout << Z << std::endl;

  return( EXIT_SUCCESS );
}


// eof - $RCSfile$
