#include "ActivationFunctions.h"
#include "NeuralNetwork.h"
#include <iostream>

int main( )
{
  using TScalar = float;
  using TAnn = NeuralNetwork< TScalar >;

  // Create a binary artifical neural network with 8 inputs.
  TAnn ann;
  ann.add( 8, 5, "identity" );
  ann.add( 3, "relu" );
  ann.add( 1, "logistic" );

  // Initialize the ANN with random weights and biases
  ann.init( );

  // Test the forward-propagation with a one-filled vector
  TAnn::TRowVector x = TAnn::TRowVector::Ones( 8 );
  std::cout
    << "Input     : "
    << x << std::endl
    << "Output    : " << ann.f( x.transpose( ) ) << std::endl
    << "Threshold : " << ann.t( x.transpose( ) ) << std::endl
    << "ANN       : " << std::endl << ann << std::endl;

  return( 0 );
}

// eof
