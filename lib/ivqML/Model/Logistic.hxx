// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Logistic__hxx__
#define __ivqML__Model__Logistic__hxx__

#include <cmath>
#include <limits>

// -------------------------------------------------------------------------
template< class _S >
template< class _X >
auto ivqML::Model::Logistic< _S >::
evaluate( const Eigen::EigenBase< _X >& iX ) const
{
  static const TScalar _0 = TScalar( 0 );
  static const TScalar _1 = TScalar( 1 );
  static const TScalar _E = std::numeric_limits< TScalar >::epsilon( );
  static const TScalar _L = std::log( _1 - _E ) - std::log( _E );
  static const auto f = [&]( TScalar z ) -> TScalar
    {
      if     ( z >  _L ) return( _1 );
      else if( z < -_L ) return( _0 );
      else               return( _1 / ( _1 + std::exp( -z ) ) );
    };

  return( this->Superclass::evaluate( iX ).unaryExpr( f ) );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _X >
auto ivqML::Model::Logistic< _S >::
threshold( const Eigen::EigenBase< _X >& iX ) const
{
  return(
    ( this->evaluate( iX ).array( ) >= TScalar( 0.5 ) )
    .template cast< TScalar >( )
    );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _G, class _X, class _Y >
typename ivqML::Model::Logistic< _S >::
TScalar ivqML::Model::Logistic< _S >::
cost(
  Eigen::EigenBase< _G >& iG,
  const Eigen::EigenBase< _X >& iX,
  const Eigen::EigenBase< _Y >& iY
  ) const
{
  /* TODO
     static const TScalar _e = std::numeric_limits< TScalar >::epsilon( );
     const auto& i = this->m_B[ b ];
     auto X =
     this->m_X->derived( )
     .block( i.first, 0, i.second, this->m_X->cols( ) )
     .template cast< TScalar >( );
     auto Y =
     this->m_Y->derived( )
     .block( i.first, 0, i.second, this->m_Y->cols( ) );

     this->m_M->operator()( this->m_Z, X );

     std::atomic< TScalar > J = 0;
     this->m_G.fill( 0 );
     auto g = this->m_G.block( 0, 1, 1, X.cols( ) );

     this->m_Z.noalias( ) =
     this->m_Z.NullaryExpr(
     this->m_Z.rows( ), this->m_Z.cols( ),
     [&]( const Eigen::Index& r, const Eigen::Index& c ) -> TScalar
     {
     TScalar z = this->m_Z( r, c );
     TScalar l = ( Y( r, 0 ) == 0 )? ( TScalar( 1 ) - z ): z;

     J  = J - std::log( ( _e < l )? l: _e );
     this->m_G( 0, 0 ) += z;
     g += X.row( r ) * z;

     return( z );
     }
     );

     J = J / TScalar( X.rows( ) );
     this->m_G /= TScalar( X.rows( ) );

     g -= this->m_Xy.row( 0 );
     this->m_G( 0, 0 ) -= this->m_Ym;

     return( std::make_pair( TScalar( J ), this->m_G.data( ) ) );
  */

  return( TScalar( 0 ) );
}

#endif // __ivqML__Model__Logistic__hxx__

// eof - $RCSfile$
