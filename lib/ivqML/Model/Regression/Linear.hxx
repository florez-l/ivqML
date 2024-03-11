// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Regression__Linear__hxx__
#define __ivqML__Model__Regression__Linear__hxx__

#include <Eigen/Dense>

// -------------------------------------------------------------------------
template< class _TScalar >
template< class _TInputX >
auto ivqML::Model::Regression::Linear< _TScalar >::
eval( const Eigen::EigenBase< _TInputX >& iX ) const
{
  auto R = this->row( this->m_P.size( ) - 1, 1 );
  auto X = iX.derived( ).template cast< TScalar >( );
  return( ( ( R * X ).array( ) + this->m_P( 0 ) ).eval( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
template< class _TInputY, class _TInputX >
void ivqML::Model::Regression::Linear< _TScalar >::
fit(
  const Eigen::EigenBase< _TInputX >& iX,
  const Eigen::EigenBase< _TInputY >& iY,
  const TScalar& l
  )
{
  auto X = iX.derived( ).template cast< TScalar >( );
  auto Y = iY.derived( ).template cast< TScalar >( );

  TNatural m = TScalar( X.cols( ) );
  TNatural n = TScalar( X.rows( ) );
  this->set_number_of_inputs( n );

  TMat Xi( m, n + 1 );
  Xi << TMat::Ones( m, 1 ), X.transpose( );

  this->row( n + 1 )
    =
    (
      Y * Xi * (
        ( ( Xi.transpose( ) * Xi ) / TScalar( m ) )
        +
        ( TMat::Identity( n + 1, n + 1 ) * l ) ).inverse( )
      )
    /
    TScalar( m );
}

#endif // __ivqML__Model__Regression__Linear__hxx__

// eof - $RCSfile$
