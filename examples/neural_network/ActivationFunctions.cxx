// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "ActivationFunctions.h"
#include <boost/algorithm/string.hpp>

// -------------------------------------------------------------------------
template< class _TScl >
const std::string& ActivationFunctions::Function< _TScl >::
name( ) const
{
  return( this->m_N );
}

// -------------------------------------------------------------------------
template< class _TScl >
const typename ActivationFunctions::Function< _TScl >::TScalar&
ActivationFunctions::Function< _TScl >::
threshold( ) const
{
  return( this->m_T );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::Function< _TScl >::
TMatrix ActivationFunctions::Function< _TScl >::
t( const TMatrix& z ) const
{
  return(
    TMatrix( ( z.array( ) >= this->m_T ).template cast< TScalar >( ) )
    );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::ArcTan< _TScl >::
TMatrix ActivationFunctions::ArcTan< _TScl >::
f( const TMatrix& z ) const
{
  return( Eigen::atan( z.array( ) ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::ArcTan< _TScl >::
TMatrix ActivationFunctions::ArcTan< _TScl >::
d( const TMatrix& z ) const
{
  return( TScalar( 1 ) / ( Eigen::pow( z.array( ), 2 ) + TScalar( 1 ) ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::BinaryStep< _TScl >::
TMatrix ActivationFunctions::BinaryStep< _TScl >::
f( const TMatrix& z ) const
{
  TMatrix r = TMatrix::Ones( z.rows( ), z.cols( ) );
  r.array( ) *= ( z.array( ) >= 0 ).template cast< TScalar >( );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::BinaryStep< _TScl >::
TMatrix ActivationFunctions::BinaryStep< _TScl >::
d( const TMatrix& z ) const
{
  return( TMatrix::Zero( z.rows( ), z.cols( ) ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::ELU< _TScl >::
TMatrix ActivationFunctions::ELU< _TScl >::
f( const TMatrix& z ) const
{
  TMatrix r( z.rows( ), z.cols( ) );
  auto c = ( z.array( ) >= 0 ).template cast< TScalar >( );
  r.array( ) =
    ( c * z.array( ) ) +
    (
      ( ( TScalar( 1 ) - c ) * this->m_Alpha ) *
      ( Eigen::exp( z.array( ) ) - TScalar( 1 ) )
      );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::ELU< _TScl >::
TMatrix ActivationFunctions::ELU< _TScl >::
d( const TMatrix& z ) const
{
  TMatrix e = this->f( z );
  TMatrix r( z.rows( ), z.cols( ) );
  auto c = ( z.array( ) < 0 ).template cast< TScalar >( );
  r.array( ) = ( c * ( e.array( ) + this->m_Alpha ) ) + ( TScalar( 1 ) - c );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::Identity< _TScl >::
TMatrix ActivationFunctions::Identity< _TScl >::
f( const TMatrix& z ) const
{
  return( z );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::Identity< _TScl >::
TMatrix ActivationFunctions::Identity< _TScl >::
d( const TMatrix& z ) const
{
  return( TMatrix::Ones( z.rows( ), z.cols( ) ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::LeakyReLU< _TScl >::
TMatrix ActivationFunctions::LeakyReLU< _TScl >::
f( const TMatrix& z ) const
{
  TMatrix r( z.rows( ), z.cols( ) );
  auto c = ( z.array( ) >= 0 ).template cast< TScalar >( );
  r.array( ) =
    ( c * z.array( ) ) +
    ( ( TScalar( 1 ) - c ) * TScalar( 1e-2 ) ) * z.array( );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::LeakyReLU< _TScl >::
TMatrix ActivationFunctions::LeakyReLU< _TScl >::
d( const TMatrix& z ) const
{
  TMatrix r( z.rows( ), z.cols( ) );
  auto c = ( z.array( ) >= 0 ).template cast< TScalar >( );
  r.array( ) = c + ( ( TScalar( 1 ) - c ) * TScalar( 1e-2 ) );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::Logistic< _TScl >::
TMatrix ActivationFunctions::Logistic< _TScl >::
f( const TMatrix& z ) const
{
  TMatrix r( z.rows( ), z.cols( ) );
  r.array( ) = TScalar( 1 ) / ( TScalar( 1 ) + Eigen::exp( -z.array( ) ) );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::Logistic< _TScl >::
TMatrix ActivationFunctions::Logistic< _TScl >::
d( const TMatrix& z ) const
{
  TMatrix e = this->f( z );
  TMatrix r( z.rows( ), z.cols( ) );
  r.array( ) = e.array( ) * ( TScalar( 1 ) - e.array( ) );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::OutTanh< _TScl >::
TMatrix ActivationFunctions::OutTanh< _TScl >::
f( const TMatrix& z ) const
{
  return( ( z.array( ).tanh( ) + TScalar( 1 ) ) / TScalar( 2 ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::OutTanh< _TScl >::
TMatrix ActivationFunctions::OutTanh< _TScl >::
d( const TMatrix& z ) const
{
  return( ( TScalar( 1 ) - z.array( ).tanh( ).square( ) ) / TScalar( 2 ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::RandomizedReLU< _TScl >::
TMatrix ActivationFunctions::RandomizedReLU< _TScl >::
f( const TMatrix& z ) const
{
  TMatrix r( z.rows( ), z.cols( ) );
  auto c = ( z.array( ) >= 0 ).template cast< TScalar >( );
  r.array( ) =
    ( c * z.array( ) ) +
    ( ( TScalar( 1 ) - c ) * this->m_Alpha ) * z.array( );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::RandomizedReLU< _TScl >::
TMatrix ActivationFunctions::RandomizedReLU< _TScl >::
d( const TMatrix& z ) const
{
  TMatrix r( z.rows( ), z.cols( ) );
  auto c = ( z.array( ) >= 0 ).template cast< TScalar >( );
  r.array( ) = c + ( ( TScalar( 1 ) - c ) * this->m_Alpha );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::ReLU< _TScl >::
TMatrix ActivationFunctions::ReLU< _TScl >::
f( const TMatrix& z ) const
{
  TMatrix r = z;
  r.array( ) *= ( z.array( ) >= 0 ).template cast< TScalar >( );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::ReLU< _TScl >::
TMatrix ActivationFunctions::ReLU< _TScl >::
d( const TMatrix& z ) const
{
  TMatrix r = TMatrix::Ones( z.rows( ), z.cols( ) );
  r.array( ) *= ( z.array( ) >= 0 ).template cast< TScalar >( );
  return( r );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::SoftPlus< _TScl >::
TMatrix ActivationFunctions::SoftPlus< _TScl >::
f( const TMatrix& z ) const
{
  return( Eigen::log( Eigen::exp( z.array( ) ) + TScalar( 1 ) ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::SoftPlus< _TScl >::
TMatrix ActivationFunctions::SoftPlus< _TScl >::
d( const TMatrix& z ) const
{
  return( TScalar( 1 ) / ( TScalar( 1 ) + Eigen::exp( -z.array( ) ) ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::Tanh< _TScl >::
TMatrix ActivationFunctions::Tanh< _TScl >::
f( const TMatrix& z ) const
{
  return( z.array( ).tanh( ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::Tanh< _TScl >::
TMatrix ActivationFunctions::Tanh< _TScl >::
d( const TMatrix& z ) const
{
  return( TScalar( 1 ) - this->f( z ).array( ).square( ) );
}

// Instances
#define _PUJ_ML_ActivationFunction_Instances( _n_, _t_ )        \
  template class ActivationFunctions::_n_< _t_ >

#define PUJ_ML_ActivationFunction_Instances( _n_ )             \
  _PUJ_ML_ActivationFunction_Instances( _n_, float );          \
  _PUJ_ML_ActivationFunction_Instances( _n_, double );         \
  _PUJ_ML_ActivationFunction_Instances( _n_, long double )

PUJ_ML_ActivationFunction_Instances( Function );
PUJ_ML_ActivationFunction_Instances( ArcTan );
PUJ_ML_ActivationFunction_Instances( BinaryStep );
PUJ_ML_ActivationFunction_Instances( ELU );
PUJ_ML_ActivationFunction_Instances( Identity );
PUJ_ML_ActivationFunction_Instances( LeakyReLU );
PUJ_ML_ActivationFunction_Instances( Logistic );
PUJ_ML_ActivationFunction_Instances( OutTanh );
PUJ_ML_ActivationFunction_Instances( RandomizedReLU );
PUJ_ML_ActivationFunction_Instances( ReLU );
PUJ_ML_ActivationFunction_Instances( SoftPlus );
PUJ_ML_ActivationFunction_Instances( Tanh );

// -------------------------------------------------------------------------
template< class _TScl >
ActivationFunctions::Factory< _TScl >::
Factory( )
{
  this->reg_cre( "arctan", &ActivationFunctions::ArcTan< _TScl >::create );
  this->reg_cre( "binarystep", &ActivationFunctions::BinaryStep< _TScl >::create );
  this->reg_cre( "elu", &ActivationFunctions::ELU< _TScl >::create );
  this->reg_cre( "identity", &ActivationFunctions::Identity< _TScl >::create );
  this->reg_cre( "leakyrelu", &ActivationFunctions::LeakyReLU< _TScl >::create );
  this->reg_cre( "logistic", &ActivationFunctions::Logistic< _TScl >::create );
  this->reg_cre( "outtanh", &ActivationFunctions::OutTanh< _TScl >::create );
  this->reg_cre( "randomizedrelu", &ActivationFunctions::RandomizedReLU< _TScl >::create );
  this->reg_cre( "relu", &ActivationFunctions::ReLU< _TScl >::create );
  this->reg_cre( "softplus", &ActivationFunctions::SoftPlus< _TScl >::create );
  this->reg_cre( "tanh", &ActivationFunctions::Tanh< _TScl >::create );
}

// -------------------------------------------------------------------------
template< class _TScl >
ActivationFunctions::Factory< _TScl >::
Factory( const Self& other )
{
}

// -------------------------------------------------------------------------
template< class _TScl >
ActivationFunctions::Factory< _TScl >::
~Factory( )
{
  this->m_FactoryMap.clear( );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::Factory< _TScl >::
Self& ActivationFunctions::Factory< _TScl >::
operator=( const Self& other )
{
  return( *this );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::Factory< _TScl >::
Self* ActivationFunctions::Factory< _TScl >::
get( )
{
  static Self instance;
  return( &instance );
}

// -------------------------------------------------------------------------
template< class _TScl >
void ActivationFunctions::Factory< _TScl >::
reg_cre( const std::string& n, TCreator c )
{
  this->m_FactoryMap[ n ] = c;
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ActivationFunctions::Factory< _TScl >::
TFunction* ActivationFunctions::Factory< _TScl >::
create( const std::string& n ) const
{
  typename TMap::const_iterator i =
    this->m_FactoryMap.find( boost::algorithm::to_lower_copy( n ) );
  if( i != this->m_FactoryMap.end( ) )
    return( i->second( ) );
  else
    return( nullptr );
}

// -------------------------------------------------------------------------
PUJ_ML_ActivationFunction_Instances( Factory );

// eof - $RCSfile$
