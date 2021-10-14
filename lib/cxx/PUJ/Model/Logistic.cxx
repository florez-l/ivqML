// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ/Model/Logistic.h>

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Logistic< _TScalar, _TTraits >::
Logistic( )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Logistic< _TScalar, _TTraits >::
Logistic( const TRow& t )
  : Superclass( t )
{
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Logistic< _TScalar, _TTraits >::
TScalar PUJ::Model::Logistic< _TScalar, _TTraits >::
operator()( const TRow& x ) const
{
  static const TScalar _0 = TScalar( 0 );
  static const TScalar _1 = TScalar( 1 );
  static const TScalar _bnd = TScalar( 40 );

  TScalar z = this->Superclass::operator()( x );
  if     ( z >  _bnd ) return( _1 );
  else if( z < -_bnd ) return( _0 );
  else                 return( _1 / ( _1 + std::exp( -z ) ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
 typename PUJ::Model::Logistic< _TScalar, _TTraits >::
TCol PUJ::Model::Logistic< _TScalar, _TTraits >::
operator()( const TMatrix& x ) const
{
  static const auto f = []( TScalar z ) -> TScalar
    {
      static const TScalar _0 = TScalar( 0 );
      static const TScalar _1 = TScalar( 1 );
      static const TScalar _bnd = TScalar( 40 );

      if     ( z >  _bnd ) return( _1 );
      else if( z < -_bnd ) return( _0 );
      else                 return( _1 / ( _1 + std::exp( -z ) ) );
    };

  return( this->Superclass::operator()( x ).unaryExpr( f ) );
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Logistic< _TScalar, _TTraits >::Cost::
Cost( Self* model, const TMatrix& X, const TCol& y, unsigned int batch_size )
  : _TBaseCost( model, X, y, batch_size )
{
  this->m_Zeros.resize( this->m_X.size( ) );
  this->m_Ones.resize( this->m_X.size( ) );

  for( unsigned int b = 0; b < this->m_X.size( ); ++b )
  {
    this->m_Zeros[ b ].clear( );
    this->m_Ones[ b ].clear( );
    PUJ::visit_lambda(
      y,
      [&]( TScalar v, int i, int j ) -> void
      {
        if( v == 0 ) this->m_Zeros[ b ].push_back( i );
        else         this->m_Ones[ b ].push_back( i );
      }
      );
  } // end for
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Logistic< _TScalar, _TTraits >::
TScalar PUJ::Model::Logistic< _TScalar, _TTraits >::Cost::
operator()( unsigned int i, TRow* g ) const
{
  static const TScalar _E = 1e-8; // std::numeric_limits< TScalar >::epsilon( );
  static const TScalar _1 = TScalar( 1 );
  unsigned long long n = this->m_X[ i ].cols( );
  unsigned long long m = this->m_X[ i ].rows( );

  TCol a = this->m_Model->operator()( this->m_X[ i ] );
  TScalar o = Eigen::log( a( this->m_Ones[ i ] ).array( ) + _E ).sum( );
  TScalar z = Eigen::log( _1 - a( this->m_Zeros[ i ] ).array( ) + _E ).sum( );

  if( g != nullptr )
  {
    if( g->cols( ) != n + 1 )
      *g = TRow::Zero( n + 1 );

    g->operator()( 0, 0 ) = a.mean( ) - this->m_uy[ i ];
    g->block( 0, 1, 1, n ) =
      ( this->m_X[ i ].array( ).colwise( ) * a.array( ) ).colwise( ).mean( ) -
      this->m_Xy[ i ].array( );
  } // end if

  return( -( o + z ) / TScalar( m ) );
}

// -------------------------------------------------------------------------
#include <PUJ_ML_export.h>

template class PUJ_ML_EXPORT PUJ::Model::Logistic< float >;
template class PUJ_ML_EXPORT PUJ::Model::Logistic< double >;

// eof - $RCSfile$
