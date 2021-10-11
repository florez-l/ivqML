// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ/Model/Linear.h>
#include <Eigen/Dense>

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::
Linear( )
{
  this->m_Parameters = TRow::Zeros( 2 );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::
Linear( const TRow& t )
{
  this->SetParameters( t );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::
Linear( const TRow& w, const TScalar& b )
{
  this->SetWeights( w );
  this->SetBias( b );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Linear< _TScalar, _TTraits >::
AnalyticalFit( const TMatrix& X, const TCol& y )
{
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
unsigned long PUJ::Model::Linear< _TScalar, _TTraits >::
GetDimensions( ) const
{
  return( this->m_Parameters.cols( ) - 1 );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const PUJ::Model::Linear< _TScalar, _TTraits >::
TRow& PUJ::Model::Linear< _TScalar, _TTraits >::
GetWeights( ) const
{
  return( this->m_Parameters.block( 0, 1, 1, this->GetDimensions( ) ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const PUJ::Model::Linear< _TScalar, _TTraits >::
TScalar& PUJ::Model::Linear< _TScalar, _TTraits >::
GetBias( ) const
{
  return( this->m_Parameters( 0, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const PUJ::Model::Linear< _TScalar, _TTraits >::
TRow& PUJ::Model::Linear< _TScalar, _TTraits >::
GetParameters( ) const
{
  return( this->m_Parameters );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Linear< _TScalar, _TTraits >::
Init( unsigned long n, const PUJ::EInitValues& e )
{
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Linear< _TScalar, _TTraits >::
SetWeights( const TRow& w )
{
  unsigned long n = this->GetDimensions( );
  if( n != w.cols( ) )
  {
    n = w.cols( );
    TScalar b = this->m_Parameters( 0, 0 );
    this->m_Parameters = TRow::Zeros( n + 1 );
    this->m_Parameters( 0, 0 ) = b;
  } // end if
  this->m_Parameters.block( 0, 1, 1, n ) = w;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Linear< _TScalar, _TTraits >::
SetBias( const TScalar& b )
{
  this->m_Parameters( 0, 0 ) = b;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Linear< _TScalar, _TTraits >::
SetParameters( const TRow& t )
{
  this->m_Parameters = t;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::
      virtual TScalar operator()( const TRow& x ) const;

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::
      virtual TCol operator()( const TMatrix& x ) const;

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::
      void _StreamIn( std::istream& i );

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::
      void _StreamOut( std::ostream& o ) const;


// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::Cost::
        Cost( Self* model, const TMatrix& X, const TCol& y );
        virtual ~Cost( ) = default;

template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::Cost::
        TScalar operator()( const TRow& t, TRow* g = nullptr ) const;































// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::
Linear( const TRow& w, const TScalar& b )
{
  this->m_Weights = w;
  this->m_Bias = b;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::
Linear( const TMatrix& X, const TCol& y )
{
  unsigned long long m = X.rows( );
  unsigned long long n = X.cols( );

  TMatrix A( n + 1, n + 1 );
  A( 0, 0 ) = 1;
  A.block( 1, 1, n, n ) = X.transpose( ) * ( X / TScalar( m ) );
  A.block( 0, 1, 1, n ) = X.colwise( ).mean( );
  A.block( 1, 0, n, 1 ) = A.block( 0, 1, 1, n ).transpose( );

  TRow d( n + 1 );
  d( 0 ) = y.mean( );
  d.block( 0, 1, 1, n ) =
    ( X.array( ).colwise( ) * y.array( ) ).colwise( ).mean( );
  TRow p = d * A.inverse( );

  this->m_Bias = p( 0, 0 );
  this->m_Weights = p.block( 0, 1, 1, n );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
unsigned long PUJ::Model::Linear< _TScalar, _TTraits >::
GetDimensions( ) const
{
  return( this->m_Weights.cols( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const typename PUJ::Model::Linear< _TScalar, _TTraits >::
TRow& PUJ::Model::Linear< _TScalar, _TTraits >::
GetWeights( ) const
{
  return( this->m_Weights );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const typename PUJ::Model::Linear< _TScalar, _TTraits >::
TScalar& PUJ::Model::Linear< _TScalar, _TTraits >::
GetBias( ) const
{
  return( this->m_Bias );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Linear< _TScalar, _TTraits >::
SetWeights( const TRow& w )
{
  this->m_Weights = w;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Linear< _TScalar, _TTraits >::
SetBias( const TScalar& b )
{
  this->m_Bias = b;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Linear< _TScalar, _TTraits >::
TScalar PUJ::Model::Linear< _TScalar, _TTraits >::
operator()( const TRow& x ) const
{
  return( ( x * this->m_Weights.transpose( ) ) + this->m_Bias );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Linear< _TScalar, _TTraits >::
TCol PUJ::Model::Linear< _TScalar, _TTraits >::
operator()( const TMatrix& x ) const
{
  return( ( x * this->m_Weights.transpose( ) ).array( ) + this->m_Bias );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Linear< _TScalar, _TTraits >::
_StreamIn( std::istream& i )
{
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Linear< _TScalar, _TTraits >::
_StreamOut( std::ostream& o ) const
{
  o << this->m_Weights.cols( ) << " "
    << this->m_Weights << " "
    << this->m_Bias;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::Cost::
Cost( Self* model, const TMatrix& X, const TCol& y )
{
  this->m_Model = model;
  this->m_XtX = ( X.transpose( ) * X ) / TScalar( X.rows( ) );
  this->m_uX = X.colwise( ).mean( );
  this->m_Xy = ( X.array( ).colwise( ) * y.array( ) ).colwise( ).mean( );
  this->m_uy = y.mean( );
  this->m_yty = ( y.transpose( ) * y )( 0, 0 ) / TScalar( X.rows( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Linear< _TScalar, _TTraits >::
TScalar PUJ::Model::Linear< _TScalar, _TTraits >::Cost::
operator()( const TRow& t, TRow* g ) const
{
  unsigned long long n = t.cols( ) - 1;
  TScalar b = t( 0, 0 );
  TRow wXtX = t.block( 0, 1, 1, n ) * this->m_XtX;
  TCol wt = t.block( 0, 1, 1, n ).transpose( );
  TScalar uXw = this->m_uX * wt;

  TScalar J =
    ( wXtX * wt )( 0, 0 ) +
    ( 2.0 * b * uXw ) +
    ( b * b ) -
    ( 2.0 * this->m_Xy * wt )( 0, 0 ) -
    ( 2.0 * b * this->m_uy ) +
    this->m_yty;

  if( g != nullptr )
  {
    if( g->cols( ) != n + 1 )
      *g = TRow::Zero( n + 1 );
    g->operator()( 0, 0 ) = uXw + b - this->m_uy;
    g->block( 0, 1, 1, n ) = wXtX + ( b * this->m_uX ) - this->m_Xy;
    *g *= 2.0;
  } // end if

  return( J );
}

// -------------------------------------------------------------------------
#include <PUJ_ML_export.h>
template class PUJ_ML_EXPORT PUJ::Model::Linear< float >;
template class PUJ_ML_EXPORT PUJ::Model::Linear< double >;

// eof - $RCSfile$
