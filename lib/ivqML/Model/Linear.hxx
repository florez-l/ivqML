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
  using _Gs = typename _G::Scalar;

  auto X = iX.derived( ).template cast< TScalar >( );
  auto Y = iY.derived( ).template cast< TScalar >( );
  TMatrix J = this->evaluate( X ) - Y.array( );

  iG.derived( )( 0, 0 ) = ( _Gs )( TScalar( 2 ) * J.mean( ) );
  iG.derived( ).block( 0, 1, 1, iG.cols( ) - 1 )
    =
    ( ( J.transpose( ) * X ) * ( TScalar( 2 ) / TScalar( X.rows( ) ) ) )
    .template cast< _Gs >( );

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
  TMatrix Xi( X.rows( ), X.cols( ) + 1 );
  Xi << TMatrix::Ones( X.rows( ), 1 ), X;

  TScalar m = TScalar( X.rows( ) );
  TNatural n = TScalar( X.cols( ) );
  this->set_number_of_inputs( n );

  TMap( this->m_T.get( ), 1, n + 1 ) =
    (
      Y.transpose( ) * Xi
      *
      (
        (
          ( Xi.transpose( ) * Xi ) / m )
        +
        ( TMatrix::Identity( n + 1, n + 1 ) * l )
        ).inverse( )
      ) / m;
}

#endif // __ivqML__Model__Linear__hxx__

// eof - $RCSfile$
