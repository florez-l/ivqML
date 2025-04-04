// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <random>
#include <ivqML/IO.h>
#include <ivqML/Model/NeuralNetwork/FeedForward.h>
#include <ivqML/Optimizer/GradientDescent.h>

int main( int argc, char** argv )
{
  std::cout << "Number of threads: " << Eigen::nbThreads( ) << std::endl;

  using TReal = long double;
  using TModel = ivqML::Model::NeuralNetwork::FeedForward< TReal >;

  TModel::TMatrix Xtr, Ltr, Xte, Lte;
  ivqML::IO::ReadMNIST( Xtr, Ltr, Xte, Lte, argv[ 1 ] );

  // Categorize labels
  TModel::TMatrix Ytr = TModel::TMatrix::Zero( 10, Ltr.cols( ) );
  for( unsigned long long c = 0; c < Ytr.cols( ); ++c )
    Ytr( Eigen::Index( Ltr( 0, c ) ), c ) = TReal( 1 );
  TModel::TMatrix Yte = TModel::TMatrix::Zero( 10, Lte.cols( ) );
  for( unsigned long long c = 0; c < Yte.cols( ); ++c )
    Yte( Eigen::Index( Lte( 0, c ) ), c ) = TReal( 1 );

  // Create model
  TModel model;
  model.set_input_size( Xtr.rows( ) );
  model.add_layer( 40, "relu" );
  model.add_layer( 20, "ReLu" );
  model.add_layer( 10, "softMax" );

  // Random generate initial parameters
  /* TODO */
  std::random_device rand_dev;
  std::mt19937 rand_gen{ rand_dev( ) };
  std::normal_distribution< TReal > rand_dist{ 0, 10 };
  model.init(
    [&rand_gen, &rand_dist]() -> TReal
    {
      return( rand_dist( rand_gen ) );
    }
    );

  using TOptimizer = ivqML::Optimizer::GradientDescent< TModel >;
  TOptimizer opt(
    Xtr.data( ), Ytr.data( ), Xte.data( ), Yte.data( ),
    Xtr.cols( ), Xte.cols( )
    );
  opt.set_batch_size( 0 );
  opt.set_regularization( 0, 0 );
  opt.set_learning_rate( 1e-2 );
  opt.set_validation_to_normal( ); // LOO, KFold
  opt.set_debugger( );
  opt.fit( &model );

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
