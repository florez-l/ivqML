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
  this->m_Dm = new TScalar[ iX.rows( ) * ( iX.cols( ) + 1 ) ];
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
ivqML::Cost::MSE< _M, _X, _Y >::
~MSE( )
{
  if( this->m_Dm != nullptr )
    delete this->m_Dm;
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
typename ivqML::Cost::MSE< _M, _X, _Y >::
TResult ivqML::Cost::MSE< _M, _X, _Y >::
operator()( ) const
{
  auto X = this->m_X->derived( ).template cast< TScalar >( );
  auto Y = this->m_Y->derived( ).template cast< TScalar >( );

  auto Ym = TMap( this->m_Ym, Y.rows( ), Y.cols( ) );
  auto Dm = TMap( this->m_Dm, X.rows( ), X.cols( ) + 1 );

  this->m_M->operator()( Ym, X );
  this->m_M->operator()( Dm, X, true );

  auto D = Ym - Y;
  TMap( this->m_G, 1, this->m_M->number_of_parameters( ) ) =
    ( Dm.array( ).colwise( ) * D.col( 0 ).array( ) )
    .colwise( ).mean( ) * TScalar( 2 );
  return( std::make_pair( D.array( ).pow( 2 ).mean( ), this->m_G ) );
}

#endif // __ivqML__Cost__MSE__hxx__

// eof - $RCSfile$
