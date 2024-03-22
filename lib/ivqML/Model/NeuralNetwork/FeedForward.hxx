// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__FeedForward__hxx__
#define __ivqML__Model__NeuralNetwork__FeedForward__hxx__

#include <cstdlib>

// -------------------------------------------------------------------------
template< class _TScl >
template< class _TInputX >
auto ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
eval( const Eigen::EigenBase< _TInputX >& iX ) const
{
  /* TODO
     TNat m = iX.cols( );
     TNat s = this->m_S.back( );
     TNat bs = this->buffer_size( ) * m;
     TRow b = TRow( bs );
     b *= TScl( 0 );

     this->_eval( iX, b.data( ) );
     return( TMat( TMatMap( b.data( ) + ( bs - ( s * m ) ), s, m ) ) );
  */
  return( iX.derived( ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
template< class _TInputX >
void ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
_eval( const Eigen::EigenBase< _TInputX >& iX, TScl* buffer ) const
{
  /* TODO
  TNat m = iX.cols( );
  TNat s = this->buffer_size( ) * m;

  // Input layer
  TMatMap( buffer, this->m_S[ 0 ], m )
    = iX.derived( ).template cast< TScl >( );

  // Loop
  TNat a = 0, z = this->m_S[ 0 ] * m;
  TNat w = 0, b = this->m_S[ 0 ] * this->m_S[ 1 ];
  TNat L = this->number_of_layers( );
  for( TNat l = 0; l < L; ++l )
  {
    TMatMap Z( buffer + z, this->m_S[ l + 1 ], m );
    TMatMap Al( buffer + a, this->m_S[ l ], m );
    TMatMap An(
      buffer + ( z + ( this->m_S[ l + 1 ] * m ) ),
      this->m_S[ l + 1 ], m
      );
    TMatCMap W( this->m_P.data( ) + w, this->m_S[ l + 1 ], this->m_S[ l ] );
    TColCMap B( this->m_P.data( ) + b, this->m_S[ l + 1 ], 1 );

    Z = ( W * Al ).colwise( ) + B;
    this->m_A[ l ].second( An, Z, false );

    if( l < L - 1 )
    {
      a = z + ( this->m_S[ l + 1 ] * m );
      z = a + ( this->m_S[ l + 1 ] * m );
      w = b + this->m_S[ l + 1 ];
      b = w + ( this->m_S[ l + 2 ] * this->m_S[ l + 1 ] );
    } // end if
  } // end for
  */
}

// -------------------------------------------------------------------------
template< class _TScl >
template< class _TInputX, class _TInputY >
void ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
backpropagation(
  TScl* G,
  TScl* B,
  const Eigen::EigenBase< _TInputX >& iX,
  const Eigen::EigenBase< _TInputY >& iY
  ) const
{
  TNat m = iX.cols( );
  TNat np = this->number_of_parameters( );
  TNat bs = this->buffer_size( ) * m;
  TNat L = this->number_of_layers( );
  TNat a = bs - ( this->m_S[ L ] * m );
  TNat z = a - ( this->m_S[ L ] * m );
  TNat b = np - this->m_S[ L ];
  TNat w = b - ( this->m_S[ L ] * this->m_S[ L - 1 ] );

  // Feed forward
  this->_eval( iX, B );

  // Last layer delta
  TMatMap( B + a, this->m_S[ L ], m ) -= iY.derived( );

  // Remaining layers
  for( TNat l = L; l > 0; --l )
  {
    TMatMap D( B + a, this->m_S[ l ], m );
    a -= ( this->m_S[ l - 1 ] + this->m_S[ l ] ) * m;
    z = a - ( this->m_S[ l - 1 ] * m );
    TMatMap E( B + a, this->m_S[ l - 1 ], m );

    // Update derivatives
    TRowMap( G + b, this->m_S[ l ], 1 ) = D.rowwise( ).mean( );
    TMatMap( G + w, this->m_S[ l ], this->m_S[ l - 1 ] )
      =
      ( D * E.transpose( ) ) / TScl( m );

    // Update delta if there is more back layers
    if( l > 1 )
    {
      b = w - this->m_S[ l - 1 ];
      w = b - ( this->m_S[ l - 1 ] * this->m_S[ l - 2 ] );

      E = this->W( l - 1 ).transpose( ) * D;

      TMatMap Z( B + z, this->m_S[ l - 1 ], m );
      this->m_A[ l - 1 ].second( Z, Z, true );
      E.array( ) *= Z.array( );
    } // end if
  } // end for
}

#endif // __ivqML__Model__NeuralNetwork__FeedForward__hxx__

// eof - $RCSfile$
