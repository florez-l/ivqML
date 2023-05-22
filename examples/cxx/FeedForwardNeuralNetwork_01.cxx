// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/Model/NeuralNetwork/FeedForward.h>

int main( int argc, char** argv )
{
  using TModel = PUJ_ML::Model::NeuralNetwork::FeedForward< double >;
  using TActivations = TModel::TBaseActivations;

  TModel model;
  model.add_layer( 2, 8, TActivations( )( "ReLU" ) );
  model.add_layer( 4, TActivations( )( "ReLU" ) );
  model.add_layer( 2, TActivations( )( "SoftMax" ) );
  model.init( );

  /* TODO
     model.init( 1 );
     model( 0 ) = 1;
     model( 1 ) = 3;

     decltype( model )::TMatrix X( 4, model.number_of_inputs( ) );
     X << 1, 2, 3, 4;
     decltype( model )::TCol Y;
     model.evaluate( Y, X );

     std::cout << Y << std::endl;
     std::cout << "===============" << std::endl;
  */

  std::cout << "===============" << std::endl;
  std::cout << model << std::endl;
  std::cout << "===============" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
