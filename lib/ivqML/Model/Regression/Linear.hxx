// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Regression__Linear__hxx__
#define __ivqML__Model__Regression__Linear__hxx__

#include <Eigen/Dense>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX >
auto ivqML::Model::Regression::Linear< _TReal, _TNatural >::
operator()( const Eigen::EigenBase< _TX >& X ) const
{
  return(
    ( ( X.derived( ).template cast< TReal >( )
        *
        TCColMap( this->m_P + 1, this->m_S - 1, 1 ) ).array( )
      +
      this->m_P[ 0 ] ).matrix( )
    );
}

// -------------------------------------------------------------------------
/**
 * TODO: Use of L1 regularization is not yet solved
 */
template< class _TReal, class _TNatural >
template< class _TX, class _Ty >
void ivqML::Model::Regression::Linear< _TReal, _TNatural >::
fit(
  const Eigen::EigenBase< _TX >& bX,
  const Eigen::EigenBase< _Ty >& by,
  const TReal& L1, const TReal& L2
  )
{
  auto X = bX.derived( ).template cast< TReal >( );
  auto y = by.derived( ).template cast< TReal >( );

  TNatural n = X.cols( );
  TNatural m = X.rows( );

  /* TODO
     if( n == 0 || m != y.rows( ) )
     throw AssertionError( 'Incompatible sizes.' )
  */

  TRow b( n + 1 );
  b( 0 ) = y.mean( );
  b.block( 0, 1, 1, n )
    =
    ( X.array( ).colwise( ) * y.col( 0 ).array( ) ).colwise( ).mean( );

  TMat A( n + 1, n + 1 );
  A( 0 , 0 ) = 1 + L2;
  A.block( 1, 1, n, n )
    =
    ( TMat::Identity( n, n ) * L2 ).array( )
    +
    ( ( X.transpose( ) * X ).array( ) / TReal( m ) );
  A.block( 0, 1, 1, n ) = X.colwise( ).mean( );
  A.block( 1, 0, n, 1 ) = A.block( 0, 1, 1, n ).transpose( );

  this->_resize( n + 1 );
  TRowMap( this->m_P, n + 1, 1 )
    =
    A.colPivHouseholderQr( ).solve( b.transpose( ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TG, class _TX, class _Ty >
typename ivqML::Model::Regression::Linear< _TReal, _TNatural >::
TReal ivqML::Model::Regression::Linear< _TReal, _TNatural >::
cost_gradient(
  Eigen::EigenBase< _TG >& G,
  const Eigen::EigenBase< _TX >& bX,
  const Eigen::EigenBase< _Ty >& by,
  const TReal& L1, const TReal& L2
  )
{
  auto X = bX.derived( ).template cast< TReal >( );
  auto y = by.derived( ).template cast< TReal >( );

  TCol z = this->operator()( X ) - y;
  G.derived( )( 0 , 0 ) = TReal( 2 ) * z.mean( );
  G.derived( ).block( 1, 0, X.cols( ), 1 )
    =
    ( X.array( ).colwise( ) * z.array( ) ).colwise( ).mean( ).transpose( );

  return( z.array( ).pow( 2 ).mean( ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX, class _Ty >
typename ivqML::Model::Regression::Linear< _TReal, _TNatural >::
TReal ivqML::Model::Regression::Linear< _TReal, _TNatural >::
cost( const Eigen::EigenBase< _TX >& X, const Eigen::EigenBase< _Ty >& y )
{
  return(
    (
      this->operator()( X.derived( ).template cast< TReal >( ) )
      -
      y.derived( ).template cast< TReal >( )
      ).array( ).pow( 2 ).mean( )
    );
}

#endif // __ivqML__Model__Regression__Linear__hxx__

// eof - $RCSfile$
