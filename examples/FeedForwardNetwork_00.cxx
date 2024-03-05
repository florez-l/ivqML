// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <ivqML/Model/NeuralNetwork/FeedForward.h>

using TReal = long double;
using TModel = ivqML::Model::NeuralNetwork::FeedForward< TReal >;

int main( int argc, char** argv )
{
  unsigned int m = 10;

  TModel model;
  model.add_layer( 8, 4, "ReLU" );
  model.add_layer( 2, "ReLU" );
  model.add_layer( 1, "siGmoID" );
  model.init( );

  std::cout << model << std::endl;

  // Some random input data
  TModel::TMatrix X( model.number_of_inputs( ), m );
  X.setRandom( );
  std::cout << "-------------- INPUTS --------------" << std::endl;
  std::cout << X << std::endl;

  std::cout << "-------------- OUTPUTS --------------" << std::endl;
  TModel::TMatrix Y = model.evaluate( X );
  std::cout << Y << std::endl;

  std::cout << "-------------- THRESHOLDS --------------" << std::endl;
  TModel::TMatrix Z = model.threshold( X );
  std::cout << Z << std::endl;

  std::cout << "-------------- COST --------------" << std::endl;
  TModel model_for_cost;
  model_for_cost = model;
  model_for_cost.init( );

  TModel::TMatrix G;
  TReal J;
  model_for_cost.cost( G, X, Y, &J );

  std::cout << "Model for cost: " << std::endl << model_for_cost << std::endl;
  std::cout << "Cost = " << J << std::endl;
  std::cout << "Gradient = " << std::endl << G.transpose( ) << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
