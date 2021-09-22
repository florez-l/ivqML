// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================


#include <iostream>




#include <PUJ/Model/Logistic.h>
#include <Eigen/Dense>
#include <cmath>

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Logistic< _TScalar, _TTraits >::
Logistic( const TRow& w, const TScalar& b )
{
  this->m_Linear = new TLinear( w, b );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Logistic< _TScalar, _TTraits >::
~Logistic( )
{
  delete this->m_Linear;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
unsigned long PUJ::Model::Logistic< _TScalar, _TTraits >::
GetDimensions( ) const
{
  return( this->m_Linear->GetDimensions( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const typename PUJ::Model::Logistic< _TScalar, _TTraits >::
TRow& PUJ::Model::Logistic< _TScalar, _TTraits >::
GetWeights( ) const
{
  return( this->m_Linear->GetWeights( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const typename PUJ::Model::Logistic< _TScalar, _TTraits >::
TScalar& PUJ::Model::Logistic< _TScalar, _TTraits >::
GetBias( ) const
{
  return( this->m_Linear->GetBias( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Logistic< _TScalar, _TTraits >::
SetWeights( const TRow& w )
{
  this->m_Linear->SetWeights( w );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Logistic< _TScalar, _TTraits >::
SetBias( const TScalar& b )
{
  this->m_Linear->SetBias( b );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Logistic< _TScalar, _TTraits >::
TScalar PUJ::Model::Logistic< _TScalar, _TTraits >::
operator()( const TRow& x, bool threshold ) const
{
  TScalar z = this->m_Linear->operator()( x );
  if( z > TScalar( 40 ) )
    return( TScalar( 1 ) );
  if( z < -TScalar( 40 ) )
    return( TScalar( 0 ) );
  else
  {
    TScalar a = TScalar( 1 ) / ( TScalar( 1 ) + std::exp( -z ) );
    if( threshold )
      return( TScalar( ( a < TScalar( 0.5 ) )? 0: 1 ) );
    else
      return( a );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Logistic< _TScalar, _TTraits >::
TCol PUJ::Model::Logistic< _TScalar, _TTraits >::
operator()( const TMatrix& x, bool threshold ) const
{
  TCol a =
    TScalar( 1 ) /
    (
      TScalar( 1 ) +
      Eigen::exp( -this->m_Linear->operator()( x ).array( ) )
      );
  if( threshold )
    return(
      TCol( ( a.array( ) >= TScalar( 0.5 ) ).template cast< TScalar >( ) )
      );
  else
    return( a );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Logistic< _TScalar, _TTraits >::Cost::
Cost( const TMatrix& X, const TCol& y )
{
  this->m_X = X;
  this->m_y = y;
  this->m_Xy = ( X.array( ).colwise( ) * y.array( ) ).colwise( ).mean( );
  this->m_uy = y.mean( );

  this->m_Zeros.clear( );
  this->m_Ones.clear( );
  PUJ::visit_lambda(
    this->m_y,
    [&]( TScalar v, int i, int j )
    {
      if( v == 0 ) this->m_Zeros.push_back( i );
      else         this->m_Ones.push_back( i );
    }
    );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Logistic< _TScalar, _TTraits >::
TScalar PUJ::Model::Logistic< _TScalar, _TTraits >::Cost::
operator()( const TRow& t, TRow* g ) const
{
  static const TScalar eps = std::numeric_limits< TScalar >::epsilon( );

  unsigned long long n = this->m_X.cols( );
  unsigned long long m = this->m_X.rows( );
  TCol a = Self( t.block( 0, 1, 1, n ), t( 0, 0 ) )( this->m_X, false );
  TScalar o = Eigen::log( eps + a( this->m_Ones ).array( ) ).sum( );
  TScalar z = Eigen::log( 1.0 + eps - a( this->m_Zeros ).array( ) ).sum( );

  if( g != nullptr )
  {
    if( g->cols( ) != n + 1 )
      *g = TRow::Zero( n + 1 );

    g->operator()( 0, 0 ) = a.mean( ) - this->m_uy;
    g->block( 0, 1, 1, n ) =
      ( this->m_X.array( ).colwise( ) * a.array( ) ).colwise( ).mean( );
  } // end if
  return( -( o + z ) / TScalar( m ) );
}

// -------------------------------------------------------------------------
#include <PUJ_ML_export.h>
template class PUJ_ML_EXPORT PUJ::Model::Logistic< float >;
template class PUJ_ML_EXPORT PUJ::Model::Logistic< double >;
template class PUJ_ML_EXPORT PUJ::Model::Logistic< long double >;

// eof - $RCSfile$
