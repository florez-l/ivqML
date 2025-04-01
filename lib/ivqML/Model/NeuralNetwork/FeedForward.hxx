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
  this->m_FwdBuf.allocate( X.cols( ), false );
  return( this->_eval( X, this->m_FwdBuf ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX, class _TY >
typename ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TReal ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
gradient(
  TReal* rG,
  const Eigen::EigenBase< _TX >& bX, const Eigen::EigenBase< _TY >& bY
  ) const
{ 
  TNatural M = bX.cols( );
  this->m_BwdBuf.allocate( M, true );

  // Forward propagation
  this->_eval( bX, this->m_BwdBuf );

  // Prepare buffers
  TNatural L = this->number_of_layers( );
  TNatural oA = ( this->m_BwdBuf.N - this->m_N[ L - 1 ] ) * M;
  TNatural oG = this->m_S - this->m_N[ L - 1 ];
  TReal* bZ = this->m_BwdBuf.Z + oA;
  TReal* bA = this->m_BwdBuf.A + oA;
  TReal* bG = rG + oG;
  TMatrixMap A( bA, this->m_N[ L - 1 ], M );
  TMatrixMap Z( bZ, this->m_N[ L - 1 ], M );
  TMatrixMap G( bG, this->m_N[ L - 1 ], 1 );

  // Backpropagate last layer
  /* TODO
     A -= bY.template cast< TReal >( );
     G = A.rowwise( ).mean( );
     bG -= this->m_N[ L - 2 ] * this->m_N[ L - 1 ];
     new ( &G ) TMatrixMap( bG, this->m_N[ L - 1 ], this->m_N[ L - 2 ] );
     G = 
  */


  return( TReal( 0 ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX >
auto ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
_eval( const Eigen::EigenBase< _TX >& bX, SBuffer& b ) const
{
  auto X = bX.derived( ).template cast< TReal >( );
  TNatural M = X.cols( );

  // Prepare forward buffer
  TReal* bZ = b.Z;
  TReal* bA = b.A;

  // Pass through first layer
  TMatrixMap Z( bZ, this->m_N[ 1 ], M );
  TMatrixMap A( bA, this->m_N[ 1 ], M );
  Z = ( this->m_W[ 0 ] * X ).colwise( ) + this->m_B[ 0 ];
  this->m_A[ 0 ].second( A, Z, false );

  // Pass through remainig layers
  for( TNatural l = 1; l < this->m_W.size( ); ++l )
  {
    if( b.K ) bZ += Z.size( );
    new ( &Z ) TMatrixMap( bZ, this->m_N[ l + 1 ], M );
    Z = ( this->m_W[ l ] * A ).colwise( ) + this->m_B[ l ];

    if( b.K ) bA += A.size( );
    new ( &A ) TMatrixMap( bA, this->m_N[ l + 1 ], M );
    this->m_A[ l ].second( A, Z, false );
  } // end for

  return( A );
}

#endif // __ivqML__Model__NeuralNetwork__FeedForward__hxx__

// eof - $RCSfile$
