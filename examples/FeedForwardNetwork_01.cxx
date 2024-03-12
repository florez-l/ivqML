// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <fstream>
#include <ivqML/Model/NeuralNetwork/FeedForward.h>
#include <ivqML/Cost/BinaryCrossEntropy.h>

using TReal = long double;
using TModel = ivqML::Model::NeuralNetwork::FeedForward< TReal >;
using TCost = ivqML::Cost::BinaryCrossEntropy< TModel >;

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

  TModel model;
  std::ifstream d_str( d.c_str( ) );
  d_str >> model;
  d_str.close( );
  std::cout << "Model: " << std::endl << model << std::endl;

  // Some random input data
  TModel::TMat X( model.number_of_inputs( ), m );
  X.setRandom( );
  std::cout << "-------------- INPUTS --------------" << std::endl;
  std::cout << X << std::endl;

  std::cout << "-------------- OUTPUTS --------------" << std::endl;
  TModel::TMat Y = model.eval( X );
  std::cout << Y << std::endl;

  std::cout << "-------------- COST --------------" << std::endl;
  TModel model_for_cost = model;
  model_for_cost.random_fill( );

  TCost J( model_for_cost );
  J.set_data( X, Y );

  TModel::TRow G( model_for_cost.number_of_parameters( ) );
  std::cout << "Model for cost: " << std::endl << model_for_cost << std::endl;
  std::cout << "Cost = " << J( G.data( ) ) << std::endl;
  std::cout << "Gradient = " << G << std::endl;

  return( EXIT_SUCCESS );
}


// eof - $RCSfile$
