// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <random>

#include <PUJ_ML/Model/NeuralNetwork/FeedForward.h>

int main( int argc, char** argv )
{
  using TModel = PUJ_ML::Model::NeuralNetwork::FeedForward< double >;
  using TActivations = TModel::TBaseActivations;

  TModel model;
  model.add_layer( 2, 16, "ReLU" );
  model.add_layer( 8, "ReLU" );
  model.add_layer( 4, "ReLU" );
  model.add_layer( 2, "SoftMax" );
  model.init( );

  TModel::TMatrix X( 10, model.number_of_inputs( ) );

  std::random_device dev;
  std::default_random_engine eng( dev( ) );
  std::uniform_real_distribution< TModel::TReal > w( -10, 10 );
  std::transform(
    X.data( ), X.data( ) + X.size( ), X.data( ),
    [&]( TModel::TReal v ) -> TModel::TReal
    {
      return( w( eng ) );
    }
    );

  TModel::TMatrix Y;
  TModel::TCol T;
  model.evaluate( Y, X );
  model.threshold( T, X );

  std::cout << "===============" << std::endl;
  std::cout << model << std::endl;
  std::cout << "===============" << std::endl;
  std::cout << Y << std::endl;
  std::cout << "===============" << std::endl;
  std::cout << T << std::endl;
  std::cout << "===============" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
