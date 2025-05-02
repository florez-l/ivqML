// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__FeedForward__hxx__
#define __ivqML__Model__NeuralNetwork__FeedForward__hxx__

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TX >
auto ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
operator()( const Eigen::EigenBase< _TX >& X ) const
{
  auto bN = this->m_N.begin( );
  auto eN = this->m_N.end( );

  TNatural M = X.cols( );
  TNatural N = X.rows( );
  TNatural As = *( std::max_element( bN, eN ) );
  TNatural Zs = *( std::max_element( ++bN, eN ) );
  TNatural Ns = ( As + Zs ) * M;

  TReal* B = reinterpret_cast< TReal* >( std::calloc( Ns, sizeof( TReal ) ) );
  TMatrixMap( B, N, M ) = X.derived( ).template cast< TReal >( );
  this->_eval( B, B + ( As * N ), M, false );
  TMatrix R = TMatrixMap( B, this->m_N.back( ), M );

  std::free( B );
  return( R );
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TX >
auto ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
threshold( const Eigen::EigenBase< _TX >& X ) const
{
  TMatrix T;
  if( this->m_A.back( ).first == "sigmoid" )
  {
  }
  else if( this->m_A.back( ).first == "softmax" )
  {
  }
  else
    T = this->operator()( X );
  return( T );
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TX, class _TY >
typename ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
TReal ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
gradient(
  TReal* bG,
  const Eigen::EigenBase< _TX >& bX, const Eigen::EigenBase< _TY >& bY,
  const TReal& l1, const TReal& l2
  ) const
{
  TNatural M = bX.cols( );

  bool own_fitting_buffer = ( this->m_FittingBuffer == nullptr );
  if( own_fitting_buffer )
    this->allocate_fitting_buffer( M );

  TNatural L = this->number_of_layers( );
  TNatural oG =  this->m_S;
  TNatural oA = std::accumulate( this->m_N.begin( ), this->m_N.end( ), 0 ) * M;
  TNatural oZ = oA - ( this->m_N[ 0 ] * M );
  TReal* A = this->m_FittingBuffer;
  TReal* Z = A + oA;
  TReal* G = bG;

  // Forward propagation
  auto Y = bY.derived( ).template cast< TReal >( );
  TMatrixMap( A, this->m_N[ 0 ], M ) = bX.derived( ).template cast< TReal >( );
  this->_eval( A, Z, M, true );

  // Compute cost
  oA -= this->m_N[ L ] * M;
  TMatrixMap D( A + oA, this->m_N[ L ], M );
  TReal J = this->m_J( Y, D );

  // Backpropagate last layer
  D -= Y;
  oG -= this->m_N[ L ];
  TMatrixMap( G + oG, this->m_N[ L ], 1 ) = D.rowwise( ).mean( );

  oG -= this->m_N[ L ] * this->m_N[ L - 1 ];
  oA -= this->m_N[ L - 1 ] * M;
  TMatrixMap E( A + oA, this->m_N[ L - 1 ], M );
  TMatrixMap( G + oG, this->m_N[ L ], this->m_N[ L - 1 ] )
    =
    ( D * E.transpose( ) ) / TReal( M );

  // Backpropagate remaining layers
  for( TNatural k = 0; k < L - 1; ++k )
  {
    TNatural l = L - k - 1;

    oZ -= this->m_N[ l ] * M;
    TMatrixMap mZ( Z + oZ, this->m_N[ l ], M );

    this->m_A[ l - 1 ].second( mZ, mZ, true );

    E = mZ.array( ) * ( this->m_W[ l ].transpose( ) * D ).array( );

    oG -= this->m_N[ l ];
    TMatrixMap( G + oG, this->m_N[ l ], 1 ) = E.rowwise( ).mean( );

    new( &D ) TMatrixMap( E.data( ), E.rows( ), E.cols( ) );
    oA -= this->m_N[ l - 1 ] * M;
    new( &E ) TMatrixMap( A + oA, this->m_N[ l - 1 ], M );

    oG -= this->m_N[ l ] * this->m_N[ l - 1 ];
    TMatrixMap( G + oG, this->m_N[ l ], this->m_N[ l - 1 ] )
      = ( D * E.transpose( ) ) / TReal( M );
  } // end for

  J = this->_R( J, bG, l1, l2 );

  if( own_fitting_buffer )
    this->free_fitting_buffer( );
  return( J );
}

#endif // __ivqML__Model__NeuralNetwork__FeedForward__hxx__

// eof - $RCSfile$
