// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <cmath>
#include <iostream>
#include <random>

#include <PUJ/Data/Algorithms.h>
#include <PUJ/Model/Logistic.h>
#include <PUJ/Optimizer/GradientDescent.h>

// -- Typedef
using TScalar    = double;
using TModel     = PUJ::Model::Logistic< TScalar >;
using TOptimizer = PUJ::Optimizer::GradientDescent< TModel >;

// -------------------------------------------------------------------------
bool debug( unsigned long long i, TScalar J, TScalar dJ, bool show )
{
  if( show )
    std::cout
      << "Iteration: " << i
      << ",  Cost = " << J << " (" << dJ << ")" << std::endl;
  return( false );
}

// -------------------------------------------------------------------------
int main( int argc, char** argv )
{
  TModel real_model( 0, 1, 0 );

  unsigned int m = 1000;
  unsigned int n = real_model.GetDimensions( );

  std::random_device dev;
  std::mt19937 gen( dev( ) );
  std::uniform_real_distribution< TScalar > dist( -0.5, 0.5 );
  TModel::TMatrix X_real =
    TModel::TMatrix::NullaryExpr(
      m, n,
      [&]( TModel::TMatrix::Index i ) -> TScalar
      {
        return( dist( gen ) );
      }
      );
  PUJ::Algorithms::Shuffle( X_real );
  TModel::TCol y_real = real_model( X_real );

  TModel opt_model;
  opt_model.Init( n, PUJ::Random );
  TModel::Cost J( &opt_model, X_real, y_real, 0 );
  TOptimizer opt( &J );
  opt.SetAlpha( 1e-4 );
  opt.SetMaximumNumberOfIterations( 100000 );
  opt.SetDebugIterations( 1000 );
  opt.SetDebug( debug );
  opt.Fit( );

  TModel::TMatrix Y_real( m, 2 );
  Y_real.block( 0, 0, m, 1 ) =
    ( y_real.array( ) >= 0.5 ).template cast< TScalar >( );
  Y_real.block( 0, 1, m, 1 ) = 1 - Y_real.block( 0, 0, m, 1 ).array( );

  TModel::TMatrix Y_estim( m, 2 );
  Y_estim.block( 0, 0, m, 1 ) =
    ( opt_model( X_real ).array( ) >= 0.5 ).template cast< TScalar >( );
  Y_estim.block( 0, 1, m, 1 ) = 1 - Y_estim.block( 0, 0, m, 1 ).array( );


  TModel::TMatrix K = Y_real.transpose( ) * Y_estim;
  TScalar acc = TScalar( 100 ) * K.diagonal( ).sum( ) / K.sum( );
  std::cout << "=======================================" << std::endl;
  std::cout << "Real model       : " << real_model << std::endl;
  std::cout << "Optimized model  : " << opt_model << std::endl;
  std::cout << "Iterations       : " << opt.GetIterations( ) << std::endl;
  std::cout << "Accuracy         : " << acc << "%" << std::endl;
  std::cout << "Confusion matrix : " << std::endl;
  std::cout << K << std::endl;
  std::cout << "=======================================" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
