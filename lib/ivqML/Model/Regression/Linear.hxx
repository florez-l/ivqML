// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Regression__Linear__hxx__
#define __ivqML__Model__Regression__Linear__hxx__

#include <Eigen/Dense>

// -------------------------------------------------------------------------
template< class _TScl >
template< class _TInputX >
auto ivqML::Model::Regression::Linear< _TScl >::
eval( const Eigen::EigenBase< _TInputX >& iX ) const
{
  return(
    ( this->m_T * iX.derived( ).template cast< TScl >( ) ).array( )
    +
    this->operator[]( 0 )
    );
}

// -------------------------------------------------------------------------
template< class _TScl >
template< class _TInputY, class _TInputX >
void ivqML::Model::Regression::Linear< _TScl >::
fit(
  const Eigen::EigenBase< _TInputX >& iX,
  const Eigen::EigenBase< _TInputY >& iY,
  const TScl& lambda
  )
{
  auto X = iX.derived( ).template cast< TScl >( );
  auto Y = iY.derived( ).template cast< TScl >( );

  TNat m = TScl( X.cols( ) );
  TNat n = TScl( X.rows( ) );
  this->set_number_of_inputs( n );

  TMat Xi( m, n + 1 );
  Xi << TMat::Ones( m, 1 ), X.transpose( );

  this->row( n + 1 )
    =
    (
      Y * Xi
      *
      (
        ( ( Xi.transpose( ) * Xi ) / TScl( m ) )
        +
        ( TMat::Identity( n + 1, n + 1 ) * lambda )
        ).inverse( )
      ) / TScl( m );
}

#endif // __ivqML__Model__Regression__Linear__hxx__

// eof - $RCSfile$
