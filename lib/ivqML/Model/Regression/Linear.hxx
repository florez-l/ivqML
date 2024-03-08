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
evaluate( const Eigen::EigenBase< _TInputX >& iX, TScalar* iB ) const
{
  auto R = this->_row( this->m_Size - 1, 1 );
  auto X = iX.derived( ).template cast< TScalar >( );
  return( ( ( R * X ).array( ) + this->m_Parameters[ 0 ] ).eval( ) );
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

  TMatrix Xi( m, n + 1 );
  Xi << TMatrix::Ones( m, 1 ), X.transpose( );

  this->_row( n + 1 )
    =
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
