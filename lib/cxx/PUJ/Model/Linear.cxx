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
Compute( TCol* g ) const
{
  if( this->m_Model == nullptr )
    return( _T( 0 ) );
  
  TCol z = ( this->m_Model->operator()( *this->m_X ) - *this->m_Y );
  _T J = z.array( ).pow( 2 ).mean( );

  if( g != nullptr )
  {
    unsigned long long n = this->m_Model->GetNumberOfParameters( );
    if( g->cols( ) != n )
      *g = TCol::Zero( n );
    g->block( 1, 0, n - 1, 1 ) =
      ( this->m_X->array( ).colwise( ) * z.array( ) ).colwise( ).mean( );
    ( *g )( 0 ) = z.mean( );
    *g *= _T( 2 );
  } // end if
  return( J );
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
Linear( const TMatrix& X, const TCol& Y )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _T >
typename PUJ_ML::Model::Linear< _T >::
TMatrix PUJ_ML::Model::Linear< _T >::
operator()( const TMatrix& x )
{
  assert( x.cols( ) + 1 == this->m_P.rows( ) );
  return(
    ( x * this->m_P.block( 1, 0, x.cols( ), 1 ) ).array( ) + this->m_P( 0 )
    );
}

// -------------------------------------------------------------------------
template class PUJ_ML_EXPORT PUJ_ML::Model::Linear< float >;
template class PUJ_ML_EXPORT PUJ_ML::Model::Linear< double >;
template class PUJ_ML_EXPORT PUJ_ML::Model::Linear< long double >;

// eof - $RCSfile$
