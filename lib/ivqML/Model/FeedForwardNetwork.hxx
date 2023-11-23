// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__FeedForwardNetwork__hxx__
#define __ivqML__Model__FeedForwardNetwork__hxx__


#include <iostream>

// -------------------------------------------------------------------------
template< class _S >
template< class _X >
auto ivqML::Model::FeedForwardNetwork< _S >::
evaluate( const Eigen::EigenBase< _X >& iX ) const
{
  return( iX.derived( ) );
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

// -------------------------------------------------------------------------
template< class _S >
template< class _X >
void ivqML::Model::FeedForwardNetwork< _S >::
_evaluate(
  const Eigen::EigenBase< _X >& iX,
  std::vector< TMatrix >& A, std::vector< TMatrix >& Z
  ) const
{
  TNatural L = this->number_of_layers( );

  if( A.size( ) != L + 1 )
  {
    A.resize( L + 1 );
    A.shrink_to_fit( );
  } // end if
  if( Z.size( ) != L )
  {
    Z.resize( L );
    Z.shrink_to_fit( );
  } // end if

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
}
*/

#endif // __ivqML__Model__FeedForwardNetwork__hxx__

// eof - $RCSfile$
