// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__FeedForwardNetwork__hxx__
#define __ivqML__Model__FeedForwardNetwork__hxx__








#include <iostream>











#include <numeric>

// -------------------------------------------------------------------------
/* TODO
template< class _S >
template< class _X >
typename ivqML::Model::FeedForwardNetwork< _S >::
TMatrix ivqML::Model::FeedForwardNetwork< _S >::
evaluate( const Eigen::EigenBase< _X >& iX ) const
{
  // Reserve some memory
  TNatural a =
    std::accumulate( this->m_S.begin( ), this->m_S.end( ), TNatural( 0 ) );
  TNatural x = ( a << 1 ) - this->m_S[ 0 ];
  std::shared_ptr< _S[] > B( new _S[ x * iX.rows( ) ] );

  // Map it to eigen world
  TNatural n = this->number_of_inputs( );
  TNatural m = iX.rows( );
  std::vector< TMap > A, Z;
  A.push_back( TMap( B.get( ), m, n ) );
  A[ 0 ] = iX.derived( ).template cast< TScalar >( );
  TNatural d = A[ 0 ].size( );
  for( TNatural l = 0; l < this->number_of_layers( ); ++l )
  {
    A.push_back( TMap( B.get( ) + d, m, this->m_S[ l + 1 ] ) );
    Z.push_back( TMap( B.get( ) + ( d << 1 ), m, this->m_S[ l + 1 ] ) );
    d += ( A.back( ).size( ) << 1 );
  } // end for

  this->_evaluate( iX, A, Z );
  TMatrix R = A.back( );
  B.reset( );
  return( R );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _G, class _X, class _Y >
void ivqML::Model::FeedForwardNetwork< _S >::
cost(
  Eigen::EigenBase< _G >& iG,
  const Eigen::EigenBase< _X >& iX,
  const Eigen::EigenBase< _Y >& iY,
  TScalar* J
  ) const
{
}

// -------------------------------------------------------------------------
template< class _S >
template< class _X >
void ivqML::Model::FeedForwardNetwork< _S >::
_evaluate(
  const Eigen::EigenBase< _X >& iX,
  std::vector< TMap >& A, std::vector< TMap >& Z
  ) const
{
  TNatural L = this->number_of_layers( );
  for( TNatural l = 0; l < L; ++l )
  {
    Z[ l ] = ( A[ l ] * this->m_W[ l ] ).rowwise( ) +  this->m_B[ l ].row( 0 );
    this->m_F[ l ].second( A[ l + 1 ], Z[ l ], false );
  } // end for
}

*/



// -------------------------------------------------------------------------
/* TODO
template< class _S >
template< class _Y, class _X >
void ivqML::Model::FeedForwardNetwork< _S >::
operator()(
  Eigen::EigenBase< _Y >& iY, const Eigen::EigenBase< _X >& iX,
  bool derivative
  ) const
{
  std::vector< TMatrix > A, Z;
  this->_eval( iX, A, Z );
  iY.derived( ) = A.back( ).template cast< typename _Y::Scalar >( );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _Y, class _X >
void ivqML::Model::FeedForwardNetwork< _S >::
backpropagate(
  const Eigen::EigenBase< _Y >& iY, const Eigen::EigenBase< _X >& iX,
  std::vector< TMatrix >& A, std::vector< TMatrix >& Z
  ) const
{
  TNatural L = this->number_of_layers( );
  this->_eval( iX, A, Z );
}
*/

#endif // __ivqML__Model__FeedForwardNetwork__hxx__

// eof - $RCSfile$
