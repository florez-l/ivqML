// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <fstream>
#include <ivqML/Model/NeuralNetwork/FeedForward.h>

using TReal = long double;
using TModel = ivqML::Model::NeuralNetwork::FeedForward< TReal >;

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
     std::cout << "---------- BACKPROPAGATION ----------" << std::endl;
     TModel::TMatrix G;
     TModel::TScalar J;
     model.init( );
     model.cost(
     G,
     TModel::TMap( X.data( ), X.rows( ), X.cols( ) ),
     TModel::TMap( Y.data( ), Y.rows( ), Y.cols( ) ),
     &J
     );
     std::cout << G << std::endl;
  */

  return( EXIT_SUCCESS );
}


// eof - $RCSfile$
