// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__FeedForward__hxx__
#define __ivqML__Model__NeuralNetwork__FeedForward__hxx__

#include <cstdlib>

// -------------------------------------------------------------------------
template< class _S >
template< class _X >
auto ivqML::Model::NeuralNetwork::FeedForward< _S >::
evaluate( const Eigen::EigenBase< _X >& iX, TScalar* iB ) const
{
  // Computation buffer
  TNatural m = iX.cols( );
  TNatural bsize = this->m_BSize * m;
  TScalar* buffer = iB;
  if( iB == nullptr )
    buffer =
      reinterpret_cast< TScalar* >(
        std::malloc( sizeof( TScalar ) * bsize )
        );

  // Loop
  TMap( buffer, this->m_S[ 0 ], m )
    = iX.derived( ).template cast< TScalar >( );

  TNatural as = 0, zs = this->m_S[ 0 ] * m;
  TNatural L = this->number_of_layers( );
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

  // Get result and free buffer
  TMatrix R = TMap( buffer + as, this->m_S[ L ], m );
  if( iB == nullptr )
    std::free( buffer );
  return( R );
}

#endif // __ivqML__Model__NeuralNetwork__FeedForward__hxx__

// eof - $RCSfile$
