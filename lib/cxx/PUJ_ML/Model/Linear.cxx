// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/Model/Linear.h>

// -------------------------------------------------------------------------
template< class _T >
PUJ_ML::Model::Linear< _T >::Cost::
Cost( Superclass* model, const TMatrix& X, const TMatrix& Y )
  : Superclass::Cost( model, X, Y )
{
}

// -------------------------------------------------------------------------
template< class _T >
_T PUJ_ML::Model::Linear< _T >::Cost::
operator()( _T* g ) const
{
  return( _T( 0 ) );
}

// -------------------------------------------------------------------------
template< class _T >
PUJ_ML::Model::Linear< _T >::
Linear( )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _T >
PUJ_ML::Model::Linear< _T >::
Linear( const TMatrix& X, const TColumn& Y )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _T >
typename PUJ_ML::Model::Linear< _T >::
TColumn PUJ_ML::Model::Linear< _T >::
operator()( const TMatrix& x )
{
  return(
    ( x * this->m_P.block( 0, 1, 1, this->m_P.cols( ) - 1 ).transpose( ) ).
    array( ) + this->m_P( 0 )
    );
}

// -------------------------------------------------------------------------
template class PUJ_ML::Model::Linear< float >;
template class PUJ_ML::Model::Linear< double >;
template class PUJ_ML::Model::Linear< long double >;

// eof - $RCSfile$
