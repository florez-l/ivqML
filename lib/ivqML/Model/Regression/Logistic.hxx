// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Regression__Logistic__hxx__
#define __ivqML__Model__Regression__Logistic__hxx__

#include <cmath>
#include <limits>

// -------------------------------------------------------------------------
template< class _TScl >
template< class _TInputX >
auto ivqML::Model::Regression::Logistic< _TScl >::
eval( const Eigen::EigenBase< _TInputX >& iX ) const
{
  static const TScl _0 = TScl( 0 );
  static const TScl _1 = TScl( 1 );
  static const TScl _E = std::numeric_limits< TScl >::epsilon( );
  static const TScl _L = std::log( _1 - _E ) - std::log( _E );

  return(
    (
      ( this->m_T * iX.derived( ).template cast< TScl >( ) ).array( )
      +
      this->operator[]( 0 )
      )
    .unaryExpr(
      []( const TScl& z ) -> TScl
      {
        if     ( z >  _L ) return( _1 );
        else if( z < -_L ) return( _0 );
        else               return( _1 / ( _1 + std::exp( -z ) ) );
      }
      )
    );
}

// -------------------------------------------------------------------------
/* TODO
   template< class _TScl >
   template< class _TInputX, class _TInputY >
   void ivqML::Model::Regression::Logistic< _TScl >::
   cost(
   TScl* bG,
   const Eigen::EigenBase< _TInputX >& iX,
   const Eigen::EigenBase< _TInputY >& iY,
   TScl* J,
   TScl* buffer
   ) const
   {
   static const TScl _E = std::numeric_limits< TScl >::epsilon( );

   auto X = iX.derived( ).template cast< TScl >( );
   auto Y = iY.derived( ).template cast< TScl >( );
   TScl m = TScl( X.cols( ) );

   TMatrix Z = this->evaluate( X );
   std::atomic< TScl > S = 0;
   Z.noalias( )
   =
   Z.NullaryExpr(
   Z.rows( ), Z.cols( ),
   [&]( const Eigen::Index& r, const Eigen::Index& c ) -> TScl
   {
   TScl z = Z( r, c );
   if( J != nullptr )
   {
   TScl l = ( Y( r, c ) == 0 )? ( TScl( 1 ) - z ): z;
   S = S - ( std::log( ( _E < l )? l: _E ) / m );
   } // end if
   return( z - Y( r, c ) );
   }
   );

   *bG = Z.mean( );
   TMap( bG + 1, iX.rows( ), 1 ) = ( X * Z.transpose( ) ) / m;
   if( J != nullptr )
   *J = TScl( S );
   }
*/

// -------------------------------------------------------------------------
template< class _TScl >
template< class _TInputX >
auto ivqML::Model::Regression::Logistic< _TScl >::
threshold( const Eigen::EigenBase< _TInputX >& iX ) const
{
  return(
    ( this->eval( iX ) >= TScl( 0.5 ) ).template cast< TScl >( )
    );
}

#endif // __ivqML__Model__Regression__Logistic__hxx__

// eof - $RCSfile$
