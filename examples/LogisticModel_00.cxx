// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <ivqML/Model/Regression/Logistic.h>
#include <ivqML/Cost/BinaryCrossEntropy.h>

using TReal = long double;
using TModel = ivqML::Model::Regression::Logistic< TReal >;
using TCost = ivqML::Cost::BinaryCrossEntropy< TModel >;

int main( int argc, char** argv )
{
  unsigned int n = 4;
  unsigned int m = 10;

  // A model
  TModel model( n );
  model.random_fill( );
  std::cout << "Model: " << model << std::endl;

  // Some random input data
  TModel::TMat X( n, m );
  X.setRandom( );
  std::cout << "-------------- INPUTS --------------" << std::endl;
  std::cout << X << std::endl;

  std::cout << "-------------- OUTPUTS --------------" << std::endl;
  TModel::TMat Y = model.eval( X );
  std::cout << Y << std::endl;

  std::cout << "-------------- COST --------------" << std::endl;
  TModel model_for_cost( model.number_of_inputs( ) );
  model_for_cost.random_fill( );

  TCost J( &model_for_cost );
  J.set_data( X, Y );

  TModel::TRow G( model_for_cost.number_of_parameters( ) );
  std::cout << "Model for cost: " << model_for_cost << std::endl;
  std::cout << "Cost = " << J( G.data( ) ) << std::endl;
  std::cout << "Gradient = " << G << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
