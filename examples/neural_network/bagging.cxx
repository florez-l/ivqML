// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "ActivationFunctions.h"
#include "NeuralNetwork.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <sstream>
#include <vector>

// -- Some typedefs
using TScalar = double; // ** WARNING **: Do not modify this!
using TAnn = NeuralNetwork< TScalar >;

// -- Helper functions
void read_files(
  TAnn::TMatrix& X, TAnn::TMatrix& Y,
  const std::string& features, const std::string& values
  );

// -- Main function
int main( int argc, char** argv )
{
  // Check inputs and get them
  if( argc < 3 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ] << " input_features.bin input_values.bin"
      << std::endl;
    return( 1 );
  } // end if
  std::string input_features = argv[ 1 ];
  std::string input_values = argv[ 2 ];

  // Bagging properties
  unsigned long gender_values = 1;
  unsigned long ethnicity_values = 6;
  unsigned int Q = 4;
  TScalar alpha = 1e-2;
  TScalar lambda = 0;

  // Read data
  TAnn::TMatrix X, Y;
  read_files( X, Y, input_features, input_values );
  unsigned int M = X.rows( );
  unsigned int N = X.cols( );
  unsigned int P = Y.cols( );

  // Extract training, testing and validation
  std::random_device rd;
  std::mt19937 gen( rd( ) );
  std::uniform_int_distribution< unsigned int > dis1( 1, 10 );

  // 1. Create a uniformly distributed column vector
  TAnn::TMatrix rand_col =
    TAnn::TMatrix::NullaryExpr(
      M, 1, [&] ( ) { return( TScalar( dis1( gen ) ) ); }
      );

  // 2. Training matrices
  auto train_idx = ( rand_col.array( ) <= 7 ).template cast< int >( );
  TAnn::TMatrix Xtrain( train_idx.sum( ), X.cols( ) );
  TAnn::TMatrix Ytrain( train_idx.sum( ), Y.cols( ) );
  unsigned int j = 0;
  for( unsigned int i = 0; i < M; ++i )
  {
    if( train_idx( i ) == 1 )
    {
      Xtrain.row( j ) = X.row( i );
      Ytrain.row( j ) = Y.row( i );
      j++;
    } // end if
  } // end for

  // 3. Testing matrices
  auto test_idx = ( rand_col.array( ) > 7 && rand_col.array( ) <= 9 ).template cast< int >( );
  TAnn::TMatrix Xtest( test_idx.sum( ), X.cols( ) );
  TAnn::TMatrix Ytest( test_idx.sum( ), Y.cols( ) );
  j = 0;
  for( unsigned int i = 0; i < M; ++i )
  {
    if( test_idx( i ) == 1 )
    {
      Xtest.row( j ) = X.row( i );
      Ytest.row( j ) = Y.row( i );
      j++;
    } // end if
  } // end for

  // 3. Validation matrices
  auto validation_idx = ( rand_col.array( ) > 9 ).template cast< int >( );
  TAnn::TMatrix Xvalidation( validation_idx.sum( ), X.cols( ) );
  TAnn::TMatrix Yvalidation( validation_idx.sum( ), Y.cols( ) );
  j = 0;
  for( unsigned int i = 0; i < M; ++i )
  {
    if( validation_idx( i ) == 1 )
    {
      Xvalidation.row( j ) = X.row( i );
      Yvalidation.row( j ) = Y.row( i );
      j++;
    } // end if
  } // end for

  // Show some info
  std::cout
    << "---------------------------" << std::endl
    << "Cross-validation sizes:" << std::endl
    << "Training   : " << Xtrain.rows( ) << std::endl
    << "Testing    : " << Xtest.rows( ) << std::endl
    << "Validation : " << Xvalidation.rows( ) << std::endl
    << "EXAMPLES   : " << M << std::endl
    << "TOTAL      : " << Xtrain.rows( ) + Xtest.rows( ) + Xvalidation.rows( )
    << std::endl
    << "---------------------------" << std::endl;

  // Prepare bagging models
  std::vector< TAnn > models( Q, TAnn( 1e-6 ) );
  unsigned int Mtrain = Xtrain.rows( );
  for( unsigned int q = 0; q < Q; ++q )
  {
    // Randomly extract examples (with replace)
    std::uniform_int_distribution< unsigned int > dis2( 0, Mtrain - 1 );
    auto indexes =
      TAnn::TMatrix::NullaryExpr(
        Mtrain, 1, [&] ( ) { return( TScalar( dis2( gen ) ) ); }
        ).template cast< unsigned int >( );
    TAnn::TMatrix Xbagg( Mtrain, Xtrain.cols( ) );
    TAnn::TMatrix Ybagg( Mtrain, Ytrain.cols( ) );
    for( unsigned int i = 0; i < Mtrain; ++i )
    {
      Xbagg.row( i ) = Xtrain.row( indexes( i, 0 ) );
      Ybagg.row( i ) = Ytrain.row( indexes( i, 0 ) );
    } // end for

    // Create neural network
    /* TODO
       models[ q ].add( Xbagg.cols( ), k1, af_1 );
       models[ q ].add( k2, af_2 );
       ...
       models[ q ].add( Ybagg.cols( ), af_out );
    */

    // Train neural network
    models[ q ].init( true );
    models[ q ].train( Xbagg, Ybagg, alpha, lambda, &std::cout );
  } // end for

  // Test bagging
  unsigned int hQ = Q >> 1; // == Q / 2
  TScalar out_thr = 0.5;
  TAnn::TMatrix Yvote = TAnn::TMatrix::Zero( Ytrain.rows( ), P );
  for( unsigned int q = 0; q < Q; ++q )
    Yvote.array( ) +=
      ( models[ q ]( Xtrain ).array( ) >= out_thr ).template cast< TScalar >( );
  TAnn::TMatrix Yfinal( Ytrain.rows( ), P );
  Yfinal.array( ) = ( Yvote.array( ) > hQ ).template cast< TScalar >( );

  // TODO: Compute F1 score from Yfinal and Ytrain

  // TODO: Validate bagging

  return( 0 );
}

// -------------------------------------------------------------------------
void read_files(
  TAnn::TMatrix& X, TAnn::TMatrix& Y,
  const std::string& features, const std::string& values
  )
{
  std::ifstream Xreader = std::ifstream( features, std::ios::binary );
  unsigned long xrows, xcols;
  Xreader.read( ( char* )( &xrows ), sizeof( unsigned long ) );
  Xreader.read( ( char* )( &xcols ), sizeof( unsigned long ) );
  X = TAnn::TMatrix::Zero( xrows, xcols );
  Xreader.read( ( char* )( X.data( ) ), sizeof( TScalar ) * xrows * xcols );
  Xreader.close( );

  // Read values
  std::ifstream Yreader = std::ifstream( values, std::ios::binary );
  unsigned long yrows, ycols;
  Yreader.read( ( char* )( &yrows ), sizeof( unsigned long ) );
  Yreader.read( ( char* )( &ycols ), sizeof( unsigned long ) );
  Y = TAnn::TMatrix::Zero( yrows, ycols );
  Yreader.read( ( char* )( Y.data( ) ), sizeof( TScalar ) * yrows * ycols );
  Yreader.close( );
}

// eof - bagging.cxx
