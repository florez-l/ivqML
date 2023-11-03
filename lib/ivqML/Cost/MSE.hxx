// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__MSE__hxx__
#define __ivqML__Cost__MSE__hxx__

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
ivqML::Cost::MSE< _M, _X, _Y >::
MSE( const _M& m, const TX& iX, const TY& iY )
  : Superclass( m, iX, iY )
{
  this->m_D = TMatrix::Zero( iX.rows( ), iX.cols( ) + 1 );
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
typename ivqML::Cost::MSE< _M, _X, _Y >::
TResult ivqML::Cost::MSE< _M, _X, _Y >::
operator()( )
{
  auto X = this->m_X->derived( ).template cast< TScalar >( );
  auto Y = this->m_Y->derived( ).template cast< TScalar >( );

  this->m_M->operator()( this->m_Z, X );
  this->m_M->operator()( this->m_D, X, true );

  auto D = ( this->m_Z - Y ).eval( );
  this->m_G.array( ) =
    ( this->m_D.array( ).colwise( ) * D.col( 0 ).array( ) )
    .colwise( ).mean( ) * TScalar( 2 );
  return( std::make_pair( D.array( ).pow( 2 ).mean( ), this->m_G.data( ) ) );
}

#endif // __ivqML__Cost__MSE__hxx__

// eof - $RCSfile$
