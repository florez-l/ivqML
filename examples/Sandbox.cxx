// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <ivqML/IO.h>
#include <ivqML/Model/NeuralNetwork/FeedForward.h>

int main( int argc, char** argv )
{
  using TReal = long double;
  using TModel = ivqML::Model::NeuralNetwork::FeedForward< TReal >;

  TModel::TMatrix Xtr, Ltr, Xte, Lte;
  ivqML::IO::ReadMNIST( Xtr, Ltr, Xte, Lte, argv[ 1 ] );

  TModel model;
  model.set_input_size( Xtr.rows( ) );
  model.add_layer( 40, "relu" );
  model.add_layer( 20, "ReLu" );
  model.add_layer( 10, "softMax" );
  model.init( );

  TModel::TMatrix Atr = model( Xtr );
  TModel::TMatrix Ate = model( Xte );

  std::cout << "----------------------------" << std::endl;
  std::cout << model << std::endl;
  std::cout << "----------------------------" << std::endl;
  std::cout << Atr.block( 0, 0, Atr.rows( ), 3 ) << std::endl;
  std::cout << "----------------------------" << std::endl;
  std::cout << Ate.block( 0, 0, Ate.rows( ), 3 ) << std::endl;
  std::cout << "----------------------------" << std::endl;


  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
