// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ/Model/Logistic.h>

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
PUJ::Model::Logistic< _TScalar, _TTraits >::
Logistic( const TRow& w, const TScalar& b )
  : Superclass( w, b )
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
Cost( const TMatrix& X, const TCol& y )
{
  this->m_Zeros.clear( );
  this->m_Ones.clear( );
  PUJ::visit_lambda(
    y,
    [&]( TScalar v, int i, int j ) -> void
    {
      if( v == 0 ) this->m_Zeros.push_back( i );
      else         this->m_Ones.push_back( i );
    }
    );

  this->m_Xy = ( X.array( ).colwise( ) * y.array( ) ).colwise( ).mean( );
  this->m_uy = y.mean( );
  this->m_X = X;
}

// -------------------------------------------------------------------------
template< class _TScalar, class _TTraits >
typename PUJ::Model::Logistic< _TScalar, _TTraits >::
TScalar PUJ::Model::Logistic< _TScalar, _TTraits >::Cost::
operator()( const TRow& t, TRow* g ) const
{
  unsigned long long n = this->m_X.cols( );
  unsigned long long m = this->m_X.rows( );

  return( TScalar( 0 ) );
    /* TODO
  static const TScalar eps = 1e-8; // std::numeric_limits< TScalar >::epsilon( );

  TCol a = Self( t.block( 0, 1, 1, n ), t( 0, 0 ) )( this->m_X );
  TScalar o = Eigen::log( a( this->m_Ones ).array( ) + eps ).sum( );
  TScalar z = Eigen::log( 1.0 - a( this->m_Zeros ).array( ) + eps ).sum( );

  if( g != nullptr )
  {
    if( g->cols( ) != n + 1 )
      *g = TRow::Zero( n + 1 );

    g->operator()( 0, 0 ) = a.mean( ) - this->m_uy;
    g->block( 0, 1, 1, n ) =
      ( this->m_X.array( ).colwise( ) * a.array( ) ).colwise( ).mean( ) -
      this->m_Xy.array( );
*/
    /* TODO
       TMatrix lll( this->m_Ones.size( ), 2 );
       lll.block( 0, 0, this->m_Ones.size( ), 1 ) = a( this->m_Ones );
       lll.block( 0, 1, this->m_Ones.size( ), 1 ) = Eigen::log( a( this->m_Ones ).array( ) + eps );

       std::cout << lll << std::endl;
       std::cout << t << std::endl;
       std::cout << o << " " << z << " " << m << " : " << *g << std::endl;
       std::exit( 1 );
    */

  // } // end if
  // TODO: return( -( o + z ) / TScalar( m ) );
}

// -------------------------------------------------------------------------
#include <PUJ_ML_export.h>
template class PUJ_ML_EXPORT PUJ::Model::Logistic< float >;
// template class PUJ_ML_EXPORT PUJ::Model::Logistic< double >;

// eof - $RCSfile$
