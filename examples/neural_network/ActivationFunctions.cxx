// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "ActivationFunctions.h"

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::Identity< _TScalar >::
TMatrix ActivationFunctions::Identity< _TScalar >::
operator()( const TMatrix& z, bool derivative ) const
{
  if( derivative )
    return( TMatrix::Ones( z.rows( ), z.cols( ) ) );
  else
    return( z );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::BinaryStep< _TScalar >::
TMatrix ActivationFunctions::BinaryStep< _TScalar >::
operator()( const TMatrix& z, bool derivative ) const
{
  if( !derivative )
  {
    TMatrix r = TMatrix::Ones( z.rows( ), z.cols( ) );
    r.array( ) *= ( z.array( ) >= 0 ).template cast< _TScalar >( );
    return( r );
  }
  else
    return( TMatrix::Zero( z.rows( ), z.cols( ) ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::Logistic< _TScalar >::
TMatrix ActivationFunctions::Logistic< _TScalar >::
operator()( const TMatrix& z, bool derivative ) const
{
  if( derivative )
  {
    TMatrix e = this->operator()( z, false );
    return( e.array( ) * ( _TScalar( 1 ) - e.array( ) ) );
  }
  else
    return( _TScalar( 1 ) / ( _TScalar( 1 ) + Eigen::exp( -z.array( ) ) ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::Tanh< _TScalar >::
TMatrix ActivationFunctions::Tanh< _TScalar >::
operator()( const TMatrix& z, bool derivative ) const
{
  if( derivative )
  {
    TMatrix t = this->operator()( z, false );
    return( _TScalar( 1 ) - Eigen::pow( t.array( ), 2 ) );
  }
  else
    return( Eigen::tanh( z.array( ) ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::ArcTan< _TScalar >::
TMatrix ActivationFunctions::ArcTan< _TScalar >::
operator()( const TMatrix& z, bool derivative ) const
{
  if( derivative )
    return( _TScalar( 1 ) / ( Eigen::pow( z.array( ), 2 ) + _TScalar( 1 ) )  );
  else
    return( Eigen::atan( z.array( ) ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::ReLU< _TScalar >::
TMatrix ActivationFunctions::ReLU< _TScalar >::
operator()( const TMatrix& z, bool derivative ) const
{
  if( derivative )
  {
    TMatrix r = TMatrix::Ones( z.rows( ), z.cols( ) );
    r.array( ) *= ( z.array( ) >= 0 ).template cast< _TScalar >( );
    return( r );
  }
  else
  {
    TMatrix r = z;
    r.array( ) *= ( z.array( ) >= 0 ).template cast< _TScalar >( );
    return( r );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::LeakyReLU< _TScalar >::
TMatrix ActivationFunctions::LeakyReLU< _TScalar >::
operator()( const TMatrix& z, bool derivative ) const
{
  if( derivative )
  {
    TMatrix r( z.rows( ), z.cols( ) );
    auto c = ( z.array( ) >= 0 ).template cast< _TScalar >( );
    r.array( ) = c + ( ( _TScalar( 1 ) - c ) * _TScalar( 1e-2 ) );
    return( r );
  }
  else
  {
    TMatrix r( z.rows( ), z.cols( ) );
    auto c = ( z.array( ) >= 0 ).template cast< _TScalar >( );
    r.array( ) =
      ( c * z.array( ) ) +
      ( ( _TScalar( 1 ) - c ) * _TScalar( 1e-2 ) ) * z.array( );
    return( r );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::RandomizedReLU< _TScalar >::
TMatrix ActivationFunctions::RandomizedReLU< _TScalar >::
operator()( const TMatrix& z, bool derivative ) const
{
  if( derivative )
  {
    TMatrix r( z.rows( ), z.cols( ) );
    auto c = ( z.array( ) >= 0 ).template cast< _TScalar >( );
    r.array( ) = c + ( ( _TScalar( 1 ) - c ) * this->m_A );
    return( r );
  }
  else
  {
    TMatrix r( z.rows( ), z.cols( ) );
    auto c = ( z.array( ) >= 0 ).template cast< _TScalar >( );
    r.array( ) =
      ( c * z.array( ) ) +
      ( ( _TScalar( 1 ) - c ) * this->m_A ) * z.array( );
    return( r );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScalar >
ActivationFunctions::RandomizedReLU< _TScalar >::
RandomizedReLU( const _TScalar& a )
{
  this->m_A = a;
}

// -------------------------------------------------------------------------
template< class _TScalar >
const _TScalar& ActivationFunctions::RandomizedReLU< _TScalar >::
GetA( ) const
{
  return( this->m_A );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::ELU< _TScalar >::
TMatrix ActivationFunctions::ELU< _TScalar >::
operator()( const TMatrix& z, bool derivative ) const
{
  if( derivative )
  {
    TMatrix e = this->operator()( z, false );
    TMatrix r( z.rows( ), z.cols( ) );
    auto c = ( z.array( ) < 0 ).template cast< _TScalar >( );
    r.array( ) = ( c * ( e.array( ) + this->m_A ) ) + ( _TScalar( 1 ) - c );
    return( r );
  }
  else
  {
    TMatrix r( z.rows( ), z.cols( ) );
    auto c = ( z.array( ) >= 0 ).template cast< _TScalar >( );
    r.array( ) =
      ( c * z.array( ) ) +
      (
        ( ( _TScalar( 1 ) - c ) * this->m_A ) *
        ( Eigen::exp( z.array( ) ) - _TScalar( 1 ) )
        );
    return( r );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScalar >
ActivationFunctions::ELU< _TScalar >::
ELU( const _TScalar& a )
{
  this->m_A = a;
}

// -------------------------------------------------------------------------
template< class _TScalar >
const _TScalar& ActivationFunctions::ELU< _TScalar >::
GetA( ) const
{
  return( this->m_A );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::SoftPlus< _TScalar >::
TMatrix ActivationFunctions::SoftPlus< _TScalar >::
operator()( const TMatrix& z, bool derivative ) const
{
  if( derivative )
    return( _TScalar( 1 ) / ( _TScalar( 1 ) + Eigen::exp( -z.array( ) ) ) );
  else
    return( Eigen::log( Eigen::exp( z.array( ) ) + _TScalar( 1 ) ) );
}

// -------------------------------------------------------------------------
#define ActivationFunction_Instances( _n_ )                     \
  template struct ActivationFunctions::_n_< float >;            \
  template struct ActivationFunctions::_n_< double >;           \
  template struct ActivationFunctions::_n_< long double >

ActivationFunction_Instances( Identity );
ActivationFunction_Instances( BinaryStep );
ActivationFunction_Instances( Logistic );
ActivationFunction_Instances( Tanh );
ActivationFunction_Instances( ArcTan );
ActivationFunction_Instances( ReLU );
ActivationFunction_Instances( LeakyReLU );
ActivationFunction_Instances( RandomizedReLU );
ActivationFunction_Instances( ELU );
ActivationFunction_Instances( SoftPlus );

// eof - $RCSfile$
