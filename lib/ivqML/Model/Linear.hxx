// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Linear__hxx__
#define __ivqML__Model__Linear__hxx__

#include <Eigen/Dense>

// -------------------------------------------------------------------------
template< class _S >
template< class _X >
auto ivqML::Model::Linear< _S >::
evaluate( const Eigen::EigenBase< _X >& iX ) const
{
  return(
    ( iX.derived( ).template cast< TScalar >( ) * this->m_nT ).array( )
    +
    this->m_T.get( )[ 0 ]
    );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _G, class _X, class _Y >
typename ivqML::Model::Linear< _S >::
TScalar ivqML::Model::Linear< _S >::
cost(
  Eigen::EigenBase< _G >& iG,
  const Eigen::EigenBase< _X >& iX,
  const Eigen::EigenBase< _Y >& iY
  ) const
{
  auto X = iX.derived( ).template cast< TScalar >( );
  auto Y = iY.derived( ).template cast< TScalar >( );

  auto J = this->evaluate( X ) - Y.array( );
  TMatrix D( X.rows( ), X.cols( ) + 1 );
  D << TMatrix::Ones( X.rows( ), 1 ), X;

  iG.derived( ) =
    (
      ( D.array( ).colwise( ) * J.col( 0 ).array( ) )
      .colwise( ).mean( ) * TScalar( 2 )
      ).template cast< typename _G::Scalar >( );

  return( J.array( ).pow( 2 ).mean( ) );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _Y, class _X >
void ivqML::Model::Linear< _S >::
fit(
  const Eigen::EigenBase< _X >& iX, const Eigen::EigenBase< _Y >& iY,
  const _S& l
  )
{
  auto X = iX.derived( ).template cast< TScalar >( );
  auto Y = iY.derived( ).template cast< TScalar >( );

  TNatural m = X.rows( );
  TNatural n = X.cols( );
  this->set_number_of_inputs( n );

  TMatrix R = TMatrix::Zero( this->m_P, this->m_P );
  R( 0, 0 ) = 1;
  R.block( 1, 1, n, n ) = ( X.transpose( ) * X ).array( ) / _S( m );
  R.block( 0, 1, 1, n ) = X.colwise( ).mean( );
  R.block( 1, 0, n, 1 ) = R.block( 0, 1, 1, n ).transpose( );

  if( l != _S( 0 ) )
  {
    /* TODO
       L = numpy.identity( n + 1 ) * l
       L[ 0 , 0 ] = 0
       R += L
    */
  } // end if

  TMatrix c = TMatrix::Zero( 1, this->m_P );
  c( 0, 0 ) = Y.mean( );
  c.block( 0, 1, 1, n ) =
    ( X.array( ).colwise( ) * Y.array( ).col( 0 ) ).colwise( ).mean( );
  this->m_nT = c * R.inverse( );
}

#endif // __ivqML__Model__Linear__hxx__

// eof - $RCSfile$
