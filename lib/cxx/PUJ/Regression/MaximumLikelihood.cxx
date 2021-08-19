// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <cassert>
#include <PUJ/Regression.h>

// -------------------------------------------------------------------------
template< class _TScalar >
PUJ::Regression::MaximumLikelihood< _TScalar >::
MaximumLikelihood( const TMatrix& X, const TMatrix& y )
  : Superclass( X, y )
{
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename PUJ::Regression::MaximumLikelihood< _TScalar >::
TScalar PUJ::Regression::MaximumLikelihood< _TScalar >::
CostAndGradient( TRowVector& gt, const TRowVector& theta )
{
}

/* TODO
   TMatrix m_Xby;
   TScalar m_uy;
   TScalar m_Eps;
*/

// -------------------------------------------------------------------------
#include <PUJ_ML_export.h>
template class PUJ_ML_EXPORT PUJ::Regression::MaximumLikelihood< float >;
template class PUJ_ML_EXPORT PUJ::Regression::MaximumLikelihood< double >;
template class PUJ_ML_EXPORT PUJ::Regression::MaximumLikelihood< long double >;

// eof - $RCSfile$
