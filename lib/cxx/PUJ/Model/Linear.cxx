// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ/Model.h>

// -------------------------------------------------------------------------
template< class _TScalar >
PUJ::Model::Linear< _TScalar >::
Linear( const TRowVector& w, const TScalar& b )
  : m_Weights( w ),
    m_Bias( b )
{
}

// -------------------------------------------------------------------------
template< class _TScalar >
unsigned long PUJ::Model::Linear< _TScalar >::
Dimensions( ) const
{
  return( this->m_Weights.cols( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
const typename PUJ::Model::Linear< _TScalar >::
TRowVector& PUJ::Model::Linear< _TScalar >::
Weights( ) const
{
  return( this->m_Weights );
}

// -------------------------------------------------------------------------
template< class _TScalar >
const typename PUJ::Model::Linear< _TScalar >::
TScalar& PUJ::Model::Linear< _TScalar >::
Bias( ) const
{
  return( this->m_Bias );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void PUJ::Model::Linear< _TScalar >::
SetWeights( const TRowVector& w )
{
  this->m_Weights = w;
}

// -------------------------------------------------------------------------
template< class _TScalar >
void PUJ::Model::Linear< _TScalar >::
SetBias( const TScalar& b )
{
  this->m_Bias = b;
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename PUJ::Model::Linear< _TScalar >::
TMatrix PUJ::Model::Linear< _TScalar >::
operator()( const TMatrix& x ) const
{
  return( ( ( this->m_Weights * x.transpose( ) ).array( ) + this->m_Bias ).transpose( ) );
}

// -------------------------------------------------------------------------
#include <PUJ_ML_export.h>
template class PUJ_ML_EXPORT PUJ::Model::Linear< float >;
template class PUJ_ML_EXPORT PUJ::Model::Linear< double >;
template class PUJ_ML_EXPORT PUJ::Model::Linear< long double >;

// eof - $RCSfile$
