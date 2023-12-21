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
  return( ( iX.derived( ) * this->m_T ).array( ) + this->m_Parameters[ 0 ] );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _G, class _X, class _Y >
void ivqML::Model::Linear< _S >::
cost(
  Eigen::EigenBase< _G >& iG,
  const Eigen::EigenBase< _X >& iX,
  const Eigen::EigenBase< _Y >& iY,
  TScalar* J
  ) const
{
  auto D = this->evaluate( iX ) - iY.derived( ).array( );

  iG.derived( )( 0, 0 ) = TScalar( 2 ) * D.mean( );
  iG.derived( ).block( 0, 1, 1, iG.cols( ) - 1 )
    =
    (
      ( D.matrix( ).transpose( ) * iX.derived( ) )
      *
      ( TScalar( 2 ) / TScalar( iX.rows( ) ) )
      );
  if( J != nullptr )
    *J = D.array( ).pow( 2 ).mean( );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _Y, class _X >
void ivqML::Model::Linear< _S >::
fit(
  const Eigen::EigenBase< _X >& iX, const Eigen::EigenBase< _Y >& iY,
  const TScalar& l
  )
{
  auto X = iX.derived( ).template cast< TScalar >( );
  auto Y = iY.derived( ).template cast< TScalar >( );

  TNatural m = TScalar( X.rows( ) );
  TNatural n = TScalar( X.cols( ) );
  this->set_number_of_inputs( n );

  TMatrix Xi( m, n + 1 );
  Xi << TMatrix::Ones( m, 1 ), X;

  TMap( this->m_Parameters.data( ), 1, n + 1 ) =
    (
      Y.transpose( ) * Xi
      *
      (
        (
          ( Xi.transpose( ) * Xi ) / m )
        +
        ( TMatrix::Identity( n + 1, n + 1 ) * l )
        ).inverse( )
      ) / TScalar( m );
}

#endif // __ivqML__Model__Linear__hxx__

// eof - $RCSfile$
