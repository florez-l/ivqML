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
using TScalar    = long double;
using TModel     = PUJ::Model::Logistic< TScalar >;
using TOptimizer = PUJ::Optimizer::GradientDescent< TScalar >;

// -------------------------------------------------------------------------
void Debugger(
  const TScalar& J, const TScalar& dJ, const TModel::TRow& t,
  unsigned long long i
  )
{
  std::cout << i << " " << J << " " << dJ << " [" << t << "]" << std::endl;
}

// -------------------------------------------------------------------------
int main( int argc, char** argv )
{
  unsigned int m = 1000;
  unsigned int n = 2;
  TModel::TRow w( n );
  w << 1, 0;
  TScalar b = 0;

  TModel real_model( w, b );

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

  TModel::Cost J( X_real, y_real );
  TOptimizer optimizer( J, n + 1 );
  optimizer.SetAlpha( 1e-2 );
  optimizer.SetMaximumNumberOfIterations( 100000 );
  optimizer.SetDebugIterations( 100 );
  optimizer.SetDebug( Debugger );
  optimizer.Fit( );

  TModel opt_model(
    optimizer.GetTheta( ).block( 0, 1, 1, n ),
    optimizer.GetTheta( )( 0, 0 )
    );
  std::cout << "=======================================" << std::endl;
  std::cout << "Real model      : " << real_model << std::endl;
  std::cout << "Optimized model : " << opt_model << std::endl;
  std::cout << "=======================================" << std::endl;

  /* TODO
     TModel::TMatrix Y_real( m, 2 );
     Y_real.block( 0, 0, m, 1 ) = y_real;
     Y_real.block( 0, 1, m, 1 ) = 1 - Y_real.block( 0, 0, m, 1 ).array( );

     TModel::TMatrix Y_estim( m, 2 );
     Y_estim.block( 0, 0, m, 1 ) = opt_model( X_real );
     Y_estim.block( 0, 1, m, 1 ) = 1 - Y_estim.block( 0, 0, m, 1 ).array( );

     std::cout << Y_real.transpose( ) * Y_estim << std::endl;
  */

  TModel::TMatrix D( m, n + 1 );
  D.block( 0, 0, m, n ) = X_real;
  D.block( 0, n, m, 1 ) = y_real;
  std::cerr << D << std::endl;



  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
