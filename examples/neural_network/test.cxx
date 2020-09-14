#include "ActivationFunctions.h"
#include "NeuralNetwork.h"
#include <iostream>

int main( )
{
  using TScalar = float;
  using TAnn = NeuralNetwork< TScalar >;

  // Create a binary artifical neural network with 8 inputs.
  TAnn ann;
  ann.add( 8, 5, ActivationFunctions::Identity< TScalar >( ) );
  ann.add( 5, 3, ActivationFunctions::ReLU< TScalar >( ) );
  ann.add( 3, 1, ActivationFunctions::Logistic< TScalar >( ) );

  // Initialize the ANN with random weights and biases
  ann.init( true );

  // Test the forward-propagation with a one-filled vector
  TAnn::TRowVector x = TAnn::TRowVector::Ones( 8 );
  TAnn::TColVector y = ann( x );

  std::cout << "Input : " << x << std::endl;
  std::cout << "Output: " << std::endl << y << std::endl;

  return( 0 );
}

// eof
