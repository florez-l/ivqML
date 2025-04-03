// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <chrono>
#include <iostream>
#include <random>
#include <ivqML/IO.h>
#include <ivqML/Model/NeuralNetwork/FeedForward.h>

#define elapsed_time( d )                                               \
  ( double( std::chrono::duration_cast< std::chrono::nanoseconds >( d ).count( ) ) * double( 1e-9 ) )

int main( int argc, char** argv )
{
  std::cout << "Number of threads: " << Eigen::nbThreads( ) << std::endl;

  using TReal = long double;
  using TModel = ivqML::Model::NeuralNetwork::FeedForward< TReal >;

  auto start = std::chrono::steady_clock::now( );
  TModel::TMatrix Xtr, Ltr, Xte, Lte;
  ivqML::IO::ReadMNIST( Xtr, Ltr, Xte, Lte, argv[ 1 ] );
  auto end = std::chrono::steady_clock::now( );
  std::cout
    << "Data read in "
    << elapsed_time( end - start )
    << " s." << std::endl;

  // Categorize labels
  start = std::chrono::steady_clock::now( );
  TModel::TMatrix Ytr = TModel::TMatrix::Zero( 10, Ltr.cols( ) );
  for( unsigned long long c = 0; c < Ytr.cols( ); ++c )
    Ytr( Eigen::Index( Ltr( 0, c ) ), c ) = TReal( 1 );
  TModel::TMatrix Yte = TModel::TMatrix::Zero( 10, Lte.cols( ) );
  for( unsigned long long c = 0; c < Yte.cols( ); ++c )
    Yte( Eigen::Index( Lte( 0, c ) ), c ) = TReal( 1 );
  end = std::chrono::steady_clock::now( );
  std::cout
    << "Data categorized in "
    << elapsed_time( end - start )
    << " s." << std::endl;

  // Create model
  start = std::chrono::steady_clock::now( );
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
  end = std::chrono::steady_clock::now( );
  std::cout
    << "Model build in "
    << elapsed_time( end - start )
    << " s." << std::endl;

  start = std::chrono::steady_clock::now( );
  TModel::TMatrix Atr = model( Xtr );
  end = std::chrono::steady_clock::now( );
  std::cout
    << "Training result computed in "
    << elapsed_time( end - start )
    << " s." << std::endl;

  start = std::chrono::steady_clock::now( );
  TModel::TMatrix Ate = model( Xte );
  end = std::chrono::steady_clock::now( );
  std::cout
    << "Testing result computed in "
    << elapsed_time( end - start )
    << " s." << std::endl;

  start = std::chrono::steady_clock::now( );
  TModel::TRow G( model.size( ) );
  G.fill( 0 );
  end = std::chrono::steady_clock::now( );
  std::cout
    << "Gradient created in "
    << elapsed_time( end - start )
    << " s." << std::endl;

  start = std::chrono::steady_clock::now( );
  TReal J = model.gradient( G.data( ), Xtr, Ytr );
  end = std::chrono::steady_clock::now( );
  std::cout
    << "1st backpropagation computed in "
    << elapsed_time( end - start )
    << " s." << std::endl;

  TReal a = 1e-4;
  start = std::chrono::steady_clock::now( );
  model -= G * a;
  end = std::chrono::steady_clock::now( );
  std::cout
    << "\t... gradient subtracted in "
    << elapsed_time( end - start )
    << " s." << std::endl;

  start = std::chrono::steady_clock::now( );
  J = model.gradient( G.data( ), Xtr, Ytr );
  end = std::chrono::steady_clock::now( );
  std::cout
    << "2nd backpropagation computed in "
    << elapsed_time( end - start )
    << " s." << std::endl;

  start = std::chrono::steady_clock::now( );
  model -= G * a;
  end = std::chrono::steady_clock::now( );
  std::cout
    << "\t... gradient subtracted in "
    << elapsed_time( end - start )
    << " s." << std::endl;

  start = std::chrono::steady_clock::now( );
  J = model.gradient( G.data( ), Xtr, Ytr );
  end = std::chrono::steady_clock::now( );
  std::cout
    << "3rd backpropagation computed in "
    << elapsed_time( end - start )
    << " s." << std::endl;

  start = std::chrono::steady_clock::now( );
  model -= G * a;
  end = std::chrono::steady_clock::now( );
  std::cout
    << "\t... gradient subtracted in "
    << elapsed_time( end - start )
    << " s." << std::endl;

  for( unsigned int i = 4; i <= 10; ++i )
  {
    start = std::chrono::steady_clock::now( );
    J = model.gradient( G.data( ), Xtr, Ytr );
    end = std::chrono::steady_clock::now( );
    std::cout
      << i << "th backpropagation computed in "
      << elapsed_time( end - start )
      << " s." << std::endl;

    start = std::chrono::steady_clock::now( );
    model -= G * a;
    end = std::chrono::steady_clock::now( );
    std::cout
      << "\t... gradient subtracted in "
      << elapsed_time( end - start )
      << " s." << std::endl;
  } // end for

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
