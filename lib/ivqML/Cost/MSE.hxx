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
  this->m_Dm =
    std::shared_ptr< TScalar[ ] >(
      new TScalar[ iX.rows( ) * ( iX.cols( ) + 1 ) ]
      );
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
ivqML::Cost::MSE< _M, _X, _Y >::
~MSE( )
{
  this->m_Dm.reset( );
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
typename ivqML::Cost::MSE< _M, _X, _Y >::
TResult ivqML::Cost::MSE< _M, _X, _Y >::
operator()( ) const
{
  auto X = this->m_X->derived( ).template cast< TScalar >( );
  auto Y = this->m_Y->derived( ).template cast< TScalar >( );

  auto Ym = TMap( this->m_Ym.get( ), Y.rows( ), Y.cols( ) );
  auto Dm = TMap( this->m_Dm.get( ), X.rows( ), X.cols( ) + 1 );

  this->m_M->operator()( Ym, X );
  this->m_M->operator()( Dm, X, true );

  auto D = Ym - Y;
  TMap( this->m_G.get( ), 1, this->m_M->number_of_parameters( ) ) =
    ( Dm.array( ).colwise( ) * D.col( 0 ).array( ) )
    .colwise( ).mean( ) * TScalar( 2 );
  return( std::make_pair( D.array( ).pow( 2 ).mean( ), this->m_G.get( ) ) );
}

#endif // __ivqML__Cost__MSE__hxx__

// eof - $RCSfile$
