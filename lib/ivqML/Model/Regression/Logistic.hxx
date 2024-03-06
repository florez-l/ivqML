// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Regression__Logistic__hxx__
#define __ivqML__Model__Regression__Logistic__hxx__

#include <cmath>
#include <limits>

// -------------------------------------------------------------------------
template< class _S >
template< class _X >
auto ivqML::Model::Regression::Logistic< _S >::
evaluate( const Eigen::EigenBase< _X >& iX ) const
{
  static const TScalar _0 = TScalar( 0 );
  static const TScalar _1 = TScalar( 1 );
  static const TScalar _E = std::numeric_limits< TScalar >::epsilon( );
  static const TScalar _L = std::log( _1 - _E ) - std::log( _E );

  return(
    this->Superclass::evaluate( iX )
    .unaryExpr(
      []( const TScalar& z ) -> TScalar
      {
        if     ( z >  _L ) return( _1 );
        else if( z < -_L ) return( _0 );
        else               return( _1 / ( _1 + std::exp( -z ) ) );
      }
      )
    );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _X, class _Y >
void ivqML::Model::Regression::Logistic< _S >::
cost(
  TScalar* bG,
  const Eigen::EigenBase< _X >& iX,
  const Eigen::EigenBase< _Y >& iY,
  TScalar* J,
  TScalar* buffer
  ) const
{
  static const TScalar _E = std::numeric_limits< TScalar >::epsilon( );

  auto X = iX.derived( ).template cast< TScalar >( );
  auto Y = iY.derived( ).template cast< TScalar >( );
  TScalar m = TScalar( X.cols( ) );

  TMatrix Z = this->evaluate( X );
  std::atomic< TScalar > S = 0;
  Z.noalias( )
    =
    Z.NullaryExpr(
      Z.rows( ), Z.cols( ),
      [&]( const Eigen::Index& r, const Eigen::Index& c ) -> TScalar
      {
        TScalar z = Z( r, c );
        if( J != nullptr )
        {
          TScalar l = ( Y( r, c ) == 0 )? ( TScalar( 1 ) - z ): z;
          S = S - ( std::log( ( _E < l )? l: _E ) / m );
        } // end if
        return( z - Y( r, c ) );
      }
      );

  *bG = Z.mean( );
  TMap( bG + 1, iX.rows( ), 1 ) = ( X * Z.transpose( ) ) / m;
  if( J != nullptr )
    *J = TScalar( S );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _X >
auto ivqML::Model::Regression::Logistic< _S >::
threshold( const Eigen::EigenBase< _X >& iX ) const
{
  return(
    ( this->evaluate( iX ) >= TScalar( 0.5 ) ).template cast< TScalar >( )
    );
}

#endif // __ivqML__Model__Regression__Logistic__hxx__

// eof - $RCSfile$
