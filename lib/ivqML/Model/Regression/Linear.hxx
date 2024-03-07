// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Regression__Linear__hxx__
#define __ivqML__Model__Regression__Linear__hxx__

#include <Eigen/Dense>

// -------------------------------------------------------------------------
template< class _S >
template< class _X >
auto ivqML::Model::Regression::Linear< _S >::
evaluate( const Eigen::EigenBase< _X >& iX ) const
{
  return( ( this->m_T * iX.derived( ) ).array( ) + this->m_Parameters[ 0 ] );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _X, class _Y >
void ivqML::Model::Regression::Linear< _S >::
cost(
  TScalar* bG,
  const Eigen::EigenBase< _X >& iX,
  const Eigen::EigenBase< _Y >& iY,
  TScalar* J,
  TScalar* buffer
  ) const
{
  static const TScalar _2 = TScalar( 2 );
  auto D = this->evaluate( iX ) - iY.derived( ).array( );

  *bG = _2 * D.mean( );
  TMap( bG + 1, iX.rows( ), 1 )
    =
    ( iX.derived( ).template cast< TScalar >( ) * D.matrix( ).transpose( ) )
    *
    ( _2 / TScalar( iX.rows( ) ) );
  if( J != nullptr )
    *J = D.array( ).pow( 2 ).mean( );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _Y, class _X >
void ivqML::Model::Regression::Linear< _S >::
fit(
  const Eigen::EigenBase< _X >& iX, const Eigen::EigenBase< _Y >& iY,
  const TScalar& l
  )
{
  auto X = iX.derived( ).template cast< TScalar >( );
  auto Y = iY.derived( ).template cast< TScalar >( );

  TNatural m = TScalar( X.cols( ) );
  TNatural n = TScalar( X.rows( ) );
  this->set_number_of_inputs( n );

  TMatrix Xi( m, n + 1 );
  Xi << TMatrix::Ones( m, 1 ), X.transpose( );

  TMap( this->m_Parameters, 1, n + 1 ) =
    (
      Y * Xi * (
        ( ( Xi.transpose( ) * Xi ) / TScalar( m ) )
        +
        ( TMatrix::Identity( n + 1, n + 1 ) * l ) ).inverse( )
      )
    /
    TScalar( m );
}

#endif // __ivqML__Model__Regression__Linear__hxx__

// eof - $RCSfile$
