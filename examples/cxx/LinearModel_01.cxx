// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <cmath>
#include <iostream>

#include <PUJ/Data/Algorithms.h>
#include <PUJ/Model/Linear.h>
#include <PUJ/Optimizer/GradientDescent.h>

// -- Typedef
using TScalar    = long double;
using TModel     = PUJ::Model::Linear< TScalar >;
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
  TScalar min_v = -10;
  TScalar max_v =  10;
  unsigned long long m = 10;
  TScalar dif_v = ( max_v - min_v ) / TScalar( m - 1 );

  unsigned long long n = 5;
  TModel::TRow w_real( n );
  w_real << 10, 20, 30, 40, 50;
  TScalar b_real = 2;
  TModel real_model( w_real, b_real );

  TModel::TMatrix X_real =
    TModel::TMatrix::NullaryExpr(
      m, n,
      [=]( TModel::TMatrix::Index i ) -> TScalar
      {
        return(
          std::pow( ( dif_v * TScalar( i % m ) ) + min_v, 1 + ( i / m ) )
          );
      }
      );
  PUJ::Algorithms::Shuffle( X_real );
  TModel::TCol y_real = real_model( X_real );

  TModel::Cost J( X_real, y_real );
  TOptimizer optimizer( J, n + 1 );
  optimizer.SetAlpha( 1e-10 );
  optimizer.SetMaximumNumberOfIterations( 100000000 );
  optimizer.SetDebugIterations( 1000000 );
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

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
