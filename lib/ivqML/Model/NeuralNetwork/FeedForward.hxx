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

// -------------------------------------------------------------------------
template< class _S >
template< class _X, class _Y >
void ivqML::Model::NeuralNetwork::FeedForward< _S >::
cost(
  TScalar* bG,
  const Eigen::EigenBase< _X >& iX,
  const Eigen::EigenBase< _Y >& iY,
  TScalar* J,
  TScalar* iB
  ) const
{
  // Computation buffer
  /* TODO
     TNatural m = iX.cols( );
     TNatural nparams = this->number_of_parameters( );
     TNatural bsize = this->m_BSize * m;
     TScalar* buffer = iB;
     if( iB == nullptr )
     buffer =
     reinterpret_cast< TScalar* >( std::malloc( sizeof( TScalar ) * bsize ) );

     // Forward propagation
     this->evaluate( iX, buffer );

     // Memory ranges
     TNatural L = this->number_of_layers( );
     TNatural as = bsize - ( this->m_S[ L ] * m );
     TNatural zs = as - ( this->m_S[ L ] * m );
     TNatural bs = nparams - this->m_S[ L ];
     TNatural ws = bs - ( this->m_S[ L ] * this->m_S[ L - 1 ] );

     // Last layer delta
     TMap( buffer + as, this->m_S[ L ], m )
     -=
     iY.derived( ).template cast< TScalar >( );

     // Remaining layers
     for( TNatural l = L; l > 0; --l )
     {
     TMap D( buffer + as, this->m_S[ l ], m );
     as -= ( this->m_S[ l - 1 ] + this->m_S[ l ] ) * m;
     zs = as - ( this->m_S[ l - 1 ] * m );
     TMap E( buffer + as, this->m_S[ l - 1 ], m );

     // Update derivatives
     TMap( bG + bs, this->m_S[ l ], 1 )
     =
     D.rowwise( ).mean( );
     TMap( bG + ws, this->m_S[ l ], this->m_S[ l - 1 ] )
     =
     ( D * E.transpose( ) ) / TScalar( m );

     // Update delta if there is more back layers
     if( l > 1 )
     {
     bs = ws - this->m_S[ l - 1 ];
     ws = bs - ( this->m_S[ l - 1 ] * this->m_S[ l - 2 ] );
     E = this->m_W[ l - 1 ].transpose( ) * D;

     TMap Z( buffer + zs, this->m_S[ l - 1 ], m );
     this->m_F[ l - 1 ].second( Z, Z, true );
     E.array( ) *= Z.array( );
     } // end if
     } // end for

     // Free buffer
     if( iB != nullptr )
     std::free( buffer );
  */
}

#endif // __ivqML__Model__NeuralNetwork__FeedForward__hxx__

// eof - $RCSfile$
