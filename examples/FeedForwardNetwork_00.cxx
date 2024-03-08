// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <ivqML/Model/NeuralNetwork/FeedForward.h>
#include <ivqML/Cost/BinaryCrossEntropy.h>

using TReal = long double;
using TModel = ivqML::Model::NeuralNetwork::FeedForward< TReal >;
using TCost = ivqML::Cost::BinaryCrossEntropy< TModel >;

int main( int argc, char** argv )
{
  unsigned int m = 10;

  TModel model;
  model.set_number_of_inputs( 8 );
  model.add_layer( 4, "ReLU" );
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

  /* TODO
     std::cout << "-------------- THRESHOLDS --------------" << std::endl;
     TModel::TMatrix Z = model.threshold( X );
     std::cout << Z << std::endl;
  */

  std::cout << "-------------- COST --------------" << std::endl;
  /* TODO
     TModel model_for_cost;
     model_for_cost.shallow_copy( model );

     TCost J( model_for_cost );
     J.set_data( X, Y );

     TModel::TRow G( model_for_cost.number_of_parameters( ) );
     std::cout << "Model for cost: " << model_for_cost << std::endl;
     std::cout << "Cost = " << J( G.data( ) ) << std::endl;
     std::cout << "Gradient = " << G << std::endl;
  */

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
