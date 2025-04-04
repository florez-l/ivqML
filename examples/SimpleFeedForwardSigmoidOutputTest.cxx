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

  // Random engine
  std::random_device rand_dev;
  std::mt19937 rand_gen{ rand_dev( ) };

  // Create model
  TModel model;
  model.set_input_size( 2 );
  model.add_layer( 4, "relu" );
  model.add_layer( 1, "SigmOid" );

  // Create some data separated by a vertical line in 0
  TModel::TNatural Mtr = 10;
  TModel::TRow Ytr( Mtr );
  std::bernoulli_distribution bern_dist( 0.5 );
  std::generate(
    Ytr.begin( ), Ytr.end( ),
    [&rand_gen, &bern_dist]() -> TReal
    {
      return( TReal( ( bern_dist( rand_gen ) )? 1: 0 ) );
    }
    );

  TModel::TNatural N = model.input_size( );
  TModel::TMatrix Xtr( N, Mtr );
  std::uniform_real_distribution< TReal > tr_dist( 1e-4, 10 );
  std::generate(
    Xtr.data( ), Xtr.data( ) + Xtr.size( ),
    [&rand_gen, &tr_dist]() -> TReal
    {
      return( tr_dist( rand_gen ) );
    }
    );
  Xtr.array( ).row( N - 1 ) *= ( Ytr.array( ) * TReal( 2 ) ) - TReal( 1 );

  // Random generate initial parameters
  /* TODO */
  std::normal_distribution< TReal > init_dist{ 0, 1 };
  model.init(
    [&rand_gen, &init_dist]() -> TReal
    {
      return( init_dist( rand_gen ) );
    }
    );

  // Fit model
  using TOptimizer = ivqML::Optimizer::GradientDescent< TModel >;
  TOptimizer opt( Xtr.data( ), Ytr.data( ), Xtr.cols( ) );
  opt.set_batch_size( 0 );
  opt.set_regularization( 0, 0 );
  opt.set_learning_rate( 1e-6 );
  opt.set_validation_to_normal( ); // LOO, KFold
  opt.set_debugger( );
  opt.fit( &model );

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
