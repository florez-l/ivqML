// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__CrossEntropy__hxx__
#define __ivqML__Cost__CrossEntropy__hxx__

#include <atomic>

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
ivqML::Cost::CrossEntropy< _M, _X, _Y >::
CrossEntropy( const _M& m, const TX& iX, const TY& iY )
  : Superclass( m, iX, iY )
{
  this->m_Ym = iY.derived( ).template cast< TScalar >( ).mean( );
  this->m_Xy =
    (
      iX.derived( ).template cast< TScalar >( ).array( ).colwise( )
      *
      iY.derived( ).template cast< TScalar >( ).col( 0 ).array( )
      ).colwise( ).mean( );
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
typename ivqML::Cost::CrossEntropy< _M, _X, _Y >::
TResult ivqML::Cost::CrossEntropy< _M, _X, _Y >::
operator()( const TNatural& b )
{
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
}

#endif // __ivqML__Cost__CrossEntropy__hxx__

// eof - $RCSfile$
