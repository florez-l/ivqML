// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__FeedForwardNetwork__hxx__
#define __ivqML__Model__FeedForwardNetwork__hxx__

// -------------------------------------------------------------------------
template< class _S >
template< class _Y, class _X >
void ivqML::Model::FeedForwardNetwork< _S >::
operator()(
  Eigen::EigenBase< _Y >& iY, const Eigen::EigenBase< _X >& iX,
  bool derivative
  ) const
{
  TNatural L = this->number_of_layers( );
  std::vector< TMatrix > A( L + 1 );
  std::vector< TMatrix > Z( L );
  A.shrink_to_fit( );
  Z.shrink_to_fit( );

  A[ 0 ] = iX.derived( ).template cast< TScalar >( );
  for( TNatural l = 0; l < L; ++l )
  {
    Z[ l ] =
      ( A[ l ] * this->m_W[ l ] ).rowwise( )
      +
      this->m_B[ l ].row( 0 );
    A[ l + 1 ] = TMatrix::Zero( Z[ l ].rows( ), Z[ l ].cols( ) );
    this->m_F[ l ].second( A[ l + 1 ], Z[ l ], false );
  } // end for

  if( derivative )
  {
    // TODO: backprop
  }
  else
    iY.derived( ) = A[ L ].template cast< typename _Y::Scalar >( );
}

#endif // __ivqML__Model__FeedForwardNetwork__hxx__

// eof - $RCSfile$
