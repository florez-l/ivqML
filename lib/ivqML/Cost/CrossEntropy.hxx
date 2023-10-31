// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__CrossEntropy__hxx__
#define __ivqML__Cost__CrossEntropy__hxx__

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
ivqML::Cost::CrossEntropy< _M, _X, _Y >::
CrossEntropy( const _M& m, const TX& iX, const TY& iY )
  : Superclass( m, iX, iY )
{
  iY.derived( ).visit( this->m_YVisitor );

  // TODO: this->m_Dm = new TScalar[ iX.rows( ) * ( iX.cols( ) + 1 ) ];
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
ivqML::Cost::CrossEntropy< _M, _X, _Y >::
~CrossEntropy( )
{
  /* TODO
     if( this->m_Dm != nullptr )
     delete this->m_Dm;
  */
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
typename ivqML::Cost::CrossEntropy< _M, _X, _Y >::
TResult ivqML::Cost::CrossEntropy< _M, _X, _Y >::
operator()( ) const
{
  auto X = this->m_X->derived( ).template cast< TScalar >( );
  auto Y = this->m_Y->derived( ).template cast< TScalar >( );
  auto Ym = TMap( this->m_Ym.get( ), Y.rows( ), Y.cols( ) );

  this->m_M->operator()( Ym, X );

  TScalar J = 0;
  J -= ( Ym( this->m_YVisitor.ones, ivq_EIGEN_ALL ).array( ) + 1e-8 ).log( ).sum( );
  J -= ( TScalar( 1 ) - Ym( this->m_YVisitor.zeros, ivq_EIGEN_ALL ).array( ) + 1e-8 ).log( ).sum( );

  auto G = TMap( this->m_G.get( ), 1, this->m_M->number_of_parameters( ) );
  G( 0, 0 ) = Ym.mean( ) - Y.mean( );
  G.block( 0, 1, 1, X.cols( ) ) =
    ( X.array( ).colwise( ) * Ym.col( 0 ).array( ) ).colwise( ).mean( )
    -
    ( X.array( ).colwise( ) * Y.col( 0 ).array( ) ).colwise( ).mean( );
  return( std::make_pair( J / TScalar( Y.rows( ) ), this->m_G.get( ) ) );
}

#endif // __ivqML__Cost__CrossEntropy__hxx__

// eof - $RCSfile$
