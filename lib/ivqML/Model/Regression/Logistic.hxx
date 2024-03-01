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
template< class _G, class _X, class _Y >
void ivqML::Model::Regression::Logistic< _S >::
cost(
  Eigen::EigenBase< _G >& iG,
  const Eigen::EigenBase< _X >& iX,
  const Eigen::EigenBase< _Y >& iY,
  TScalar* J
  ) const
{
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

// -------------------------------------------------------------------------
/* TODO
   cost(
   Eigen::EigenBase< _G >& iG,
   const Eigen::EigenBase< _X >& iX,
   const Eigen::EigenBase< _Y >& iY,
   TScalar* J
   ) const
   {
   using _Gs = typename _G::Scalar;

   static const TScalar _e = std::numeric_limits< TScalar >::epsilon( );

   auto X = iX.derived( ).template cast< TScalar >( );
   auto Y = iY.derived( ).template cast< TScalar >( );
   TScalar m = TScalar( X.rows( ) );

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
   S = S - ( std::log( ( _e < l )? l: _e ) / m );
   } // end if
   return( z - Y( r, c ) );
   }
   );

   iG.derived( )( 0, 0 ) = _Gs( Z.mean( ) );
   iG.derived( ).block( 0, 1, 1, iG.cols( ) - 1 )
   =
   ( ( Z.transpose( ) * X ) / m ).template cast< _Gs >( );

   if( J != nullptr )
   *J = TScalar( S );
   }
*/

#endif // __ivqML__Model__Regression__Logistic__hxx__

// eof - $RCSfile$
