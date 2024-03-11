// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Regression/Logistic.h>

// -------------------------------------------------------------------------
template< class _TScalar >
ivqML::Model::Regression::Logistic< _TScalar >::
Logistic( const TNatural& n )
  : Superclass( n )
{
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Regression::Logistic< float >;
template class ivqML_EXPORT ivqML::Model::Regression::Logistic< double >;
template class ivqML_EXPORT ivqML::Model::Regression::Logistic< long double >;

// eof - $RCSfile$
