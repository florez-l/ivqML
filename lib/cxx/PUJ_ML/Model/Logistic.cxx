// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/Model/Logistic.h>
#include <cmath>

// -------------------------------------------------------------------------
template< class _T >
PUJ_ML::Model::Logistic< _T >::Cost::
Cost( Superclass* model, const TMatrix& X, const TMatrix& Y )
  : Superclass::Cost( model, X, Y )
{
}

// -------------------------------------------------------------------------
template< class _T >
_T PUJ_ML::Model::Logistic< _T >::Cost::
operator()( _T* g ) const
{
  return( _T( 0 ) );
}

// -------------------------------------------------------------------------
template< class _T >
PUJ_ML::Model::Logistic< _T >::
Logistic( )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _T >
PUJ_ML::Model::Logistic< _T >::
Logistic( const TMatrix& X, const TCol& Y )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _T >
typename PUJ_ML::Model::Logistic< _T >::
TMatrix PUJ_ML::Model::Logistic< _T >::
operator()( const TMatrix& x )
{
  static const auto f = []( _T z ) -> _T
    {
      static const _T _0 = _T( 0 );
      static const _T _1 = _T( 1 );
      static const _T _b = _T( 40 );

      if     ( z >  _b ) return( _1 );
      else if( z < -_b ) return( _0 );
      else               return( _1 / ( _1 + std::exp( -z ) ) );
    };
  return( this->Superclass::operator()( x ).unaryExpr( f ) );
}

// -------------------------------------------------------------------------
template< class _T >
typename PUJ_ML::Model::Logistic< _T >::
TMatrix PUJ_ML::Model::Logistic< _T >::
operator[]( const TMatrix& x )
{
  return(
    ( this->operator()( x ).array( ) >= _T( 0.5 ) ).template cast< _T >( )
    );
}

// -------------------------------------------------------------------------
template class PUJ_ML_EXPORT PUJ_ML::Model::Logistic< float >;
template class PUJ_ML_EXPORT PUJ_ML::Model::Logistic< double >;
template class PUJ_ML_EXPORT PUJ_ML::Model::Logistic< long double >;

// eof - $RCSfile$
