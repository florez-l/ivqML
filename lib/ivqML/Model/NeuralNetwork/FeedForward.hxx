// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__FeedForward__hxx__
#define __ivqML__Model__NeuralNetwork__FeedForward__hxx__

// -------------------------------------------------------------------------
template< class _S >
template< class _X >
auto ivqML::Model::NeuralNetwork::FeedForward< _S >::
evaluate( const Eigen::EigenBase< _X >& iX ) const
{
  // Computation buffer
  TNatural m = iX.cols( );
  TNatural bsize = ( ( this->m_BSize << 1 ) - this->m_S[ 0 ] ) * m;
  std::vector< TScalar > data( bsize, TScalar( 0 ) );
  data.shrink_to_fit( );
  TScalar* buffer = data.data( );

  // Loop
  TMap( buffer, this->m_S[ 0 ], m )
    = iX.derived( ).template cast< TScalar >( );

  TNatural as = 0, zs = this->m_S[ 0 ] * m;
  TNatural L = this->m_W.size( );
  for( TNatural l = 0; l < L; ++l )
  {
    TMap Z( buffer + zs, this->m_S[ l + 1 ], m );
    TMap Al( buffer + as, this->m_S[ l ], m );
    TMap An(
      buffer + ( zs + ( this->m_S[ l + 1 ] * m ) ), this->m_S[ l + 1 ], m
      );

    Z = ( this->m_W[ l ] * Al ).colwise( ) + this->m_B[ l ].col( 0 );
    this->m_F[ l ].second( An, Z, false );

    as += ( this->m_S[ l ] + this->m_S[ l + 1 ] ) * m;
    zs += ( this->m_S[ l + 1 ] + this->m_S[ l + 1 ] ) * m;
  } // end for

  return( TMatrix( TMap( buffer + as, this->m_S[ L ], m ) ) );
}

/* TODO
   template< class _S >
   template< class _G, class _X, class _Y >
   void ivqML::Model::FeedForward< _S >::
   cost(
   Eigen::EigenBase< _G >& iG,
   const Eigen::EigenBase< _X >& iX,
   const Eigen::EigenBase< _Y >& iY,
   TScalar* J
   ) const
   {
   }
*/
/* TODO
   template< class _S >
   template< class _Y, class _X >
   void ivqML::Model::FeedForward< _S >::
   backpropagate(
   const Eigen::EigenBase< _Y >& iY, const Eigen::EigenBase< _X >& iX,
   std::vector< TMatrix >& A, std::vector< TMatrix >& Z
   ) const
   {
   TNatural L = this->number_of_layers( );
   this->_eval( iX, A, Z );
   }
*/

#endif // __ivqML__Model__NeuralNetwork__FeedForward__hxx__

// eof - $RCSfile$
