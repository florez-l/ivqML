// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Logistic.h>

// -------------------------------------------------------------------------
template< class _S >
ivqML::Model::Logistic< _S >::
Logistic( const TNatural& n )
  : Superclass( n )
{
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Logistic< float >;
template class ivqML_EXPORT ivqML::Model::Logistic< double >;
template class ivqML_EXPORT ivqML::Model::Logistic< long double >;

// eof - $RCSfile$
