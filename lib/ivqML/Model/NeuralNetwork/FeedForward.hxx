// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__FeedForward__hxx__
#define __ivqML__Model__NeuralNetwork__FeedForward__hxx__

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX >
auto ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
operator()( const Eigen::EigenBase< _TX >& X ) const
{
  this->m_FwdBuf.allocate( this->m_N, X.cols( ), false );
  this->m_FwdBuf.A[ 0 ] = X.derived( ).template cast< TReal >( );
  this->_eval( this->m_FwdBuf );
  return( this->m_FwdBuf.A.back( ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX, class _TY >
typename ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TReal ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
gradient(
  TReal* bG,
  const Eigen::EigenBase< _TX >& bX, const Eigen::EigenBase< _TY >& bY
  ) const
{
  TReal* G = bG;
  auto X = bX.derived( ).template cast< TReal >( );
  auto Y = bY.derived( ).template cast< TReal >( );

  // Prepare buffers
  TNatural M = X.cols( );
  this->m_BwdBuf.allocate( this->m_N, M, true );

  // Forward propagation
  this->m_BwdBuf.A[ 0 ] = X;
  this->_eval( this->m_BwdBuf );

  // Compute cost
  TNatural L = this->number_of_layers( );
  TReal J = this->m_J( Y, this->m_BwdBuf.A[ L ] );

  // Backpropagate last layer
  this->m_BwdBuf.A[ L ] -= Y;
  TNatural oG =  this->m_S - this->m_N[ L ];
  TMatrixMap( G + oG, this->m_N[ L ], 1 )
    =
    this->m_BwdBuf.A[ L ].rowwise( ).mean( );
  oG -= this->m_N[ L ] * this->m_N[ L - 1 ];
  TMatrixMap( G + oG, this->m_N[ L ], this->m_N[ L - 1 ] )
    =
    ( this->m_BwdBuf.A[ L ] * this->m_BwdBuf.A[ L - 1 ].transpose( ) )
    /
    TReal( M );

  // Backpropagate remaining layers
  for( TNatural k = 0; k < L - 1; ++k )
  {
    TNatural l = L - k - 1;
    this->m_A[ l - 1 ]
      .second( this->m_BwdBuf.Z[ l - 1 ], this->m_BwdBuf.Z[ l - 1 ], true );

    this->m_BwdBuf.A[ l ].array( )
      =
      this->m_BwdBuf.Z[ l - 1 ].array( )
      *
      ( this->m_W[ l ] * this->m_BwdBuf.A[ l + 1 ] ).array( );

    oG -= this->m_N[ l ];
    TMatrixMap( G + oG, this->m_N[ l ], 1 )
      =
      this->m_BwdBuf.A[ l ].rowwise( ).mean( );

    oG -= this->m_N[ l ] * this->m_N[ l - 1 ];
    TMatrixMap( G + oG, this->m_N[ l ], this->m_N[ l - 1 ] )
      =
      ( this->m_BwdBuf.A[ l ] * this->m_BwdBuf.A[ l - 1 ].transpose( ) )
      /
      TReal( M );
  } // end for

  return( J );
}

#endif // __ivqML__Model__NeuralNetwork__FeedForward__hxx__

// eof - $RCSfile$
