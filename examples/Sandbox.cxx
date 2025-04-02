// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <random>
#include <ivqML/IO.h>
#include <ivqML/Model/NeuralNetwork/FeedForward.h>

int main( int argc, char** argv )
{
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
  std::random_device rand_dev;
  std::mt19937 rand_gen{ rand_dev( ) };
  std::normal_distribution< TReal > rand_dist{ 0, 1e-2 };
  model.init(
    [&rand_gen, &rand_dist]() -> TReal
    {
      return( rand_dist( rand_gen ) );
    }
    );

  TModel::TMatrix Atr = model( Xtr );
  TModel::TMatrix Ate = model( Xte );

  std::cout << "----------------------------" << std::endl;
  std::cout << model << std::endl;
  std::cout << "----------------------------" << std::endl;
  std::cout << Atr.block( 0, 0, Atr.rows( ), 3 ) << std::endl;
  std::cout << "----------------------------" << std::endl;
  std::cout << Ate.block( 0, 0, Ate.rows( ), 3 ) << std::endl;
  std::cout << "----------------------------" << std::endl;


  TModel::TRow G( model.size( ) );
  G.fill( 0 );
  TReal J = model.gradient( G.data( ), Xtr, Ytr );

  /* TODO
     std::cout << G << std::endl;
     std::cout << "----------------------------" << std::endl;
  */
  std::cout << J << std::endl;
  std::cout << "----------------------------" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
