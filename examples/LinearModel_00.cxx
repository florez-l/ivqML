// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <ivqML/Model/Regression/Linear.h>

using TReal = long double;
using TModel = ivqML::Model::Regression::Linear< TReal >;

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

  std::cout << "-------------- COST --------------" << std::endl;
  TModel model_for_cost( n );
  model_for_cost.random_fill( );

  TModel::TMatrix G;
  TReal J;
  model_for_cost.cost( G, X, Y, &J );

  std::cout << "Model for cost: " << model_for_cost << std::endl;
  std::cout << "Cost = " << J << std::endl;
  std::cout << "Gradient = " << std::endl << G << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
