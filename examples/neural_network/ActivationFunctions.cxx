// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "ActivationFunctions.h"

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::Identity< _TScalar >::
TColVector ActivationFunctions::Identity< _TScalar >::
operator()( const TColVector& z ) const
{
  return( z );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::Identity< _TScalar >::
TColVector ActivationFunctions::Identity< _TScalar >::
operator[]( const TColVector& z ) const
{
  return( TColVector::Ones( z.rows( ), z.cols( ) ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::BinaryStep< _TScalar >::
TColVector ActivationFunctions::BinaryStep< _TScalar >::
operator()( const TColVector& z ) const
{
  TColVector r = TColVector::Ones( z.rows( ), z.cols( ) );
  r.array( ) *= ( z.array( ) >= 0 ).template cast< _TScalar >( );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::BinaryStep< _TScalar >::
TColVector ActivationFunctions::BinaryStep< _TScalar >::
operator[]( const TColVector& z ) const
{
  return( TColVector::Zero( z.rows( ), z.cols( ) ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::Logistic< _TScalar >::
TColVector ActivationFunctions::Logistic< _TScalar >::
operator()( const TColVector& z ) const
{
  return( _TScalar( 1 ) / ( _TScalar( 1 ) + Eigen::exp( -z.array( ) ) ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::Logistic< _TScalar >::
TColVector ActivationFunctions::Logistic< _TScalar >::
operator[]( const TColVector& z ) const
{
  TColVector e = this->operator()( z );
  return( e.array( ) * ( _TScalar( 1 ) - e.array( ) ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::Tanh< _TScalar >::
TColVector ActivationFunctions::Tanh< _TScalar >::
operator()( const TColVector& z ) const
{
  return( Eigen::tanh( z.array( ) ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::Tanh< _TScalar >::
TColVector ActivationFunctions::Tanh< _TScalar >::
operator[]( const TColVector& z ) const
{
  TColVector t = this->operator()( z );
  return( _TScalar( 1 ) - Eigen::pow( t.array( ), 2 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::ArcTan< _TScalar >::
TColVector ActivationFunctions::ArcTan< _TScalar >::
operator()( const TColVector& z ) const
{
  return( Eigen::atan( z.array( ) ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::ArcTan< _TScalar >::
TColVector ActivationFunctions::ArcTan< _TScalar >::
operator[]( const TColVector& z ) const
{
  return( _TScalar( 1 ) / ( Eigen::pow( z.array( ), 2 ) + _TScalar( 1 ) )  );
}


// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::ReLU< _TScalar >::
TColVector ActivationFunctions::ReLU< _TScalar >::
operator()( const TColVector& z ) const
{
  TColVector r = z;
  r.array( ) *= ( z.array( ) >= 0 ).template cast< _TScalar >( );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::ReLU< _TScalar >::
TColVector ActivationFunctions::ReLU< _TScalar >::
operator[]( const TColVector& z ) const
{
  TColVector r = TColVector::Ones( z.rows( ), z.cols( ) );
  r.array( ) *= ( z.array( ) >= 0 ).template cast< _TScalar >( );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::LeakyReLU< _TScalar >::
TColVector ActivationFunctions::LeakyReLU< _TScalar >::
operator()( const TColVector& z ) const
{
  TColVector r( z.rows( ), z.cols( ) );
  auto c = ( z.array( ) >= 0 ).template cast< _TScalar >( );
  r.array( ) =
    ( c * z.array( ) ) +
    ( ( _TScalar( 1 ) - c ) * _TScalar( 1e-2 ) ) * z.array( );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::LeakyReLU< _TScalar >::
TColVector ActivationFunctions::LeakyReLU< _TScalar >::
operator[]( const TColVector& z ) const
{
  TColVector r( z.rows( ), z.cols( ) );
  auto c = ( z.array( ) >= 0 ).template cast< _TScalar >( );
  r.array( ) = c + ( ( _TScalar( 1 ) - c ) * _TScalar( 1e-2 ) );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::RandomizedReLU< _TScalar >::
TColVector ActivationFunctions::RandomizedReLU< _TScalar >::
operator()( const TColVector& z ) const
{
  TColVector r( z.rows( ), z.cols( ) );
  auto c = ( z.array( ) >= 0 ).template cast< _TScalar >( );
  r.array( ) =
    ( c * z.array( ) ) +
    ( ( _TScalar( 1 ) - c ) * this->m_A ) * z.array( );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::RandomizedReLU< _TScalar >::
TColVector ActivationFunctions::RandomizedReLU< _TScalar >::
operator[]( const TColVector& z ) const
{
  TColVector r( z.rows( ), z.cols( ) );
  auto c = ( z.array( ) >= 0 ).template cast< _TScalar >( );
  r.array( ) = c + ( ( _TScalar( 1 ) - c ) * this->m_A );
  return( r );
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
TColVector ActivationFunctions::ELU< _TScalar >::
operator()( const TColVector& z ) const
{
  TColVector r( z.rows( ), z.cols( ) );
  auto c = ( z.array( ) >= 0 ).template cast< _TScalar >( );
  r.array( ) =
    ( c * z.array( ) ) +
    (
      ( ( _TScalar( 1 ) - c ) * this->m_A ) *
      ( Eigen::exp( z.array( ) ) - _TScalar( 1 ) )
      );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::ELU< _TScalar >::
TColVector ActivationFunctions::ELU< _TScalar >::
operator[]( const TColVector& z ) const
{
  TColVector e = this->operator()( z );
  TColVector r( z.rows( ), z.cols( ) );
  auto c = ( z.array( ) < 0 ).template cast< _TScalar >( );
  r.array( ) = ( c * ( e.array( ) + this->m_A ) ) + ( _TScalar( 1 ) - c );
  return( r );
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
TColVector ActivationFunctions::SoftPlus< _TScalar >::
operator()( const TColVector& z ) const
{
  return( Eigen::log( Eigen::exp( z.array( ) ) + _TScalar( 1 ) ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ActivationFunctions::SoftPlus< _TScalar >::
TColVector ActivationFunctions::SoftPlus< _TScalar >::
operator[]( const TColVector& z ) const
{
  return( _TScalar( 1 ) / ( _TScalar( 1 ) + Eigen::exp( -z.array( ) ) ) );
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
