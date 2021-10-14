// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <fstream>
#include <iostream>
#include <sstream>

#include <PUJ/Data/Algorithms.h>
#include <PUJ/Model/Logistic.h>
#include <PUJ/Optimizer/GradientDescent.h>

// -- Typedef
using TScalar    = long double;
using TModel     = PUJ::Model::Logistic< TScalar >;
using TOptimizer = PUJ::Optimizer::GradientDescent< TScalar >;

// -------------------------------------------------------------------------
bool debug( unsigned long long i, TScalar J, bool show )
{
  if( show )
    std::cout << "Iteration: " << i << ",  Cost = " << J << std::endl;
  return( false );
}

// -------------------------------------------------------------------------
int main( int argc, char** argv )
{
  if( argc < 2 )
  {
    std::cerr << "Usage: " << argv[ 0 ] << " input" << std::endl;
    return( EXIT_FAILURE );
  } // end if

  std::ifstream input( argv[ 1 ] );
  std::stringstream buffer;
  buffer << input.rdbuf( );
  input.close( );

  std::string line;
  std::vector< TScalar > raw;
  unsigned long long m = 0;
  while( std::getline( buffer, line ) )
  {
    std::replace( line.begin( ), line.end( ), ',', ' ' );
    std::istringstream line_str( line );
    TScalar v;
    while( line_str >> v )
      raw.push_back( v );
    m++;
  } // end while

  Eigen::Map< TModel::TMatrix > D( raw.data( ), raw.size( ) / m, m );
  TModel::TMatrix X_real = D.block( 0, 0, D.rows( ) - 1, m ).transpose( );
  TModel::TCol y_real = D.block( D.rows( ) - 1, 0, 1, m ).transpose( );
  unsigned long long n = X_real.cols( );

  TModel::Cost J( X_real, y_real );
  TOptimizer optimizer( J, n + 1 );
  optimizer.SetAlpha( 1e-4 );
  optimizer.SetMaximumNumberOfIterations( 100000 );
  optimizer.SetDebugIterations( 10000 );
  optimizer.SetDebug( Debugger );
  optimizer.Fit( );

  /* TODO
     TModel opt_model(
     optimizer.GetTheta( ).block( 0, 1, 1, n ),
     optimizer.GetTheta( )( 0, 0 )
     );
     std::cout << "=======================================" << std::endl;
     std::cout << "Optimized model : " << opt_model << std::endl;
     std::cout << "=======================================" << std::endl;

     TModel::TMatrix Y_real( m, 2 );
     Y_real.block( 0, 0, m, 1 ) = y_real;
     Y_real.block( 0, 1, m, 1 ) = 1 - Y_real.block( 0, 0, m, 1 ).array( );

     TModel::TMatrix Y_estim( m, 2 );
     Y_estim.block( 0, 0, m, 1 ) = opt_model( X_real );
     Y_estim.block( 0, 1, m, 1 ) = 1 - Y_estim.block( 0, 0, m, 1 ).array( );

     TModel::TMatrix K = Y_real.transpose( ) * Y_estim;
     std::cout << K << std::endl;
     std::cout
     << "Accuracy: "
     << 100 * K.diagonal( ).sum( ) / K.sum( ) << "%"
     << std::endl;
  */

  /* TODO
     TModel::TMatrix D( m, n + 1 );
     D.block( 0, 0, m, n ) = X_real;
     D.block( 0, n, m, 1 ) = y_real;
     std::cerr << D << std::endl;
  */

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
