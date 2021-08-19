// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <cassert>
#include <PUJ/Regression.h>

// -------------------------------------------------------------------------
template< class _TScalar >
PUJ::Regression::Base< _TScalar >::
Base( const TMatrix& X, const TMatrix& y )
  : m_X( X ),
    m_y( y )
{
  assert( this->m_X.rows( ) == this->m_y.rows( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
unsigned long PUJ::Regression::Base< _TScalar >::
NumberOfExamples( ) const
{
  return( this->m_X.rows( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
unsigned long PUJ::Regression::Base< _TScalar >::
VectorSize( ) const
{
  return( this->m_X.cols( ) + 1 );
}

// -------------------------------------------------------------------------
#include <PUJ_ML_export.h>
template class PUJ_ML_EXPORT PUJ::Regression::Base< float >;
template class PUJ_ML_EXPORT PUJ::Regression::Base< double >;
template class PUJ_ML_EXPORT PUJ::Regression::Base< long double >;

// eof - $RCSfile$
