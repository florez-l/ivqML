// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ/Model/Linear.h>

#include <random>
#include <Eigen/Dense>

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::
Linear( )
  : m_W( nullptr )
{
  this->SetParameters( TRow::Zero( 1 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::
Linear( const TRow& t )
  : m_W( nullptr )
{
  this->SetParameters( t );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::
~Linear( )
{
  delete this->m_W;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Linear< _TScalar, _TTraits >::
AnalyticalFit( const TMatrix& X, const TCol& y )
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
  this->SetParameters( d * A.inverse( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const unsigned long& PUJ::Model::Linear< _TScalar, _TTraits >::
GetDimensions( ) const
{
  return( this->m_N );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Linear< _TScalar, _TTraits >::
TRow PUJ::Model::Linear< _TScalar, _TTraits >::
GetWeights( ) const
{
  return( this->m_W->transpose( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Linear< _TScalar, _TTraits >::
TScalar PUJ::Model::Linear< _TScalar, _TTraits >::
GetBias( ) const
{
  return( this->m_Parameters( 0, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
const typename PUJ::Model::Linear< _TScalar, _TTraits >::
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
  if( e == PUJ::Zeros )
    this->SetParameters( TRow::Zero( n + 1 ) );
  else if( e == PUJ::Ones )
    this->SetParameters( TRow::Ones( n + 1 ) );
  else // if( e == PUJ::Random )
  {
    std::random_device dev;
    std::mt19937 g( dev( ) );
    std::uniform_real_distribution< TScalar > d( -1, 1 );
    this->SetParameters(
      TRow::NullaryExpr(
        n + 1,
        [&]( typename TRow::Index i ) -> TScalar { return( d( g ) ); }
        )
      );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Linear< _TScalar, _TTraits >::
SetWeights( const TRow& w )
{
  if( this->m_N != w.cols( ) )
  {
    TRow t = TRow::Zero( w.cols( ) + 1 );
    t( 0, 0 ) = this->m_Parameters( 0, 0 );
    t.block( 0, 1, 1, w.cols( ) ) = w;
    this->SetParameters( t );
  }
  else
    this->m_Parameters.block( 0, 1, 1, this->m_N ) = w;
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
  this->m_N = this->m_Parameters.cols( ) - 1;
  if( this->m_W != nullptr )
    delete this->m_W;
  this->m_W = new _TCol( this->m_Parameters.data( ) + 1, this->m_N );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Linear< _TScalar, _TTraits >::
TScalar PUJ::Model::Linear< _TScalar, _TTraits >::
operator()( const TRow& x ) const
{
  return( ( x * *( this->m_W ) ) + this->m_Parameters( 0, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Linear< _TScalar, _TTraits >::
TCol PUJ::Model::Linear< _TScalar, _TTraits >::
operator()( const TMatrix& x ) const
{
  return( ( x * *( this->m_W ) ).array( ) + this->m_Parameters( 0, 0 ) );
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
  o << this->m_Parameters.cols( ) - 1 << " " << this->m_Parameters;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Linear< _TScalar, _TTraits >::Cost::
Cost( Self* model, const TMatrix& X, const TCol& y, unsigned int batch_size )
  : _TBaseCost( model, X, y, batch_size )
{
  for( unsigned int i = 0; i < this->m_X.size( ); ++i )
  {
    this->m_XtX.push_back(
      ( this->m_X[ i ].transpose( ) * this->m_X[ i ] ) /
      TScalar( this->m_X[ i ].rows( ) )
      );
    this->m_uX.push_back( this->m_X[ i ].colwise( ).mean( ) );
    this->m_Xy.push_back(
      (
        this->m_X[ i ].array( ).colwise( ) * this->m_Y[ i ].col( 0 ).array( )
        ).colwise( ).mean( )
      );
    this->m_uy.push_back( this->m_Y[ i ].mean( ) );
    this->m_yty.push_back(
      this->m_Y[ i ].squaredNorm( ) / TScalar( this->m_X[ i ].rows( ) )
      );
  } // end for
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Linear< _TScalar, _TTraits >::
TScalar PUJ::Model::Linear< _TScalar, _TTraits >::Cost::
operator()( unsigned int i, TRow* g ) const
{
  TScalar b = this->m_Model->GetBias( );
  TRow w = this->m_Model->GetWeights( );
  TRow wXtX = w * this->m_XtX[ i ];
  TCol wt = w.transpose( );
  TScalar uXw = ( this->m_uX[ i ] * wt )( 0, 0 );
  TScalar J =
    ( wXtX * wt )( 0, 0 ) +
    ( 2.0 * b * uXw ) +
    ( b * b ) -
    ( 2.0 * this->m_Xy[ i ] * wt )( 0, 0 ) -
    ( 2.0 * b * this->m_uy[ i ] ) +
    this->m_yty[ i ];

  if( g != nullptr )
  {
    unsigned long n = this->m_Model->GetDimensions( );
    if( g->cols( ) != n + 1 )
      *g = TRow::Zero( n + 1 );
    g->operator()( 0, 0 ) = uXw + b - this->m_uy[ i ];
    g->block( 0, 1, 1, n ) = wXtX + ( b * this->m_uX[ i ] ) - this->m_Xy[ i ];
    *g *= 2.0;
  } // end if

  return( J );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
void PUJ::Model::Linear< _TScalar, _TTraits >::Cost::
operator-=( const TRow& g )
{
  this->m_Model->m_Parameters -= g;
}

// -------------------------------------------------------------------------
#include <PUJ_ML_export.h>

template class PUJ_ML_EXPORT PUJ::Model::Linear< float >;
template class PUJ_ML_EXPORT PUJ::Model::Linear< double >;

// eof - $RCSfile$
