// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <cmath>
#include <iostream>

#include <PUJ/Data/Algorithms.h>
#include <PUJ/Model/Linear.h>

// -- Typedef
using TScalar = double;
using TModel = PUJ::Model::Linear< TScalar >;

int main( int argc, char** argv )
{
  unsigned long long n = 1;
  if( argc > 1 )
    n = std::atoi( argv[ 1 ] );

  TScalar min_v = -10;
  TScalar max_v =  10;
  unsigned long long m = 1000;
  TScalar dif_v = ( max_v - min_v ) / TScalar( m - 1 );

  TModel real_model;
  real_model.Init( n, PUJ::Random );

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

  TModel analytical_model;
  analytical_model.AnalyticalFit( X_real, y_real );
  TModel::TCol y_diff = y_real - analytical_model( X_real );

  std::cout << "=======================================" << std::endl;
  std::cout << "Real model       : " << real_model << std::endl;
  std::cout << "Analytical model : " << analytical_model << std::endl;
  std::cout << "Difference      : " << y_diff.norm( ) << std::endl;
  std::cout << "=======================================" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
