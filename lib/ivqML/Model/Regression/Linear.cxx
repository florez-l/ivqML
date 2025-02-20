// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Regression/Linear.h>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
ivqML::Model::Regression::Linear< _TReal, _TNatural >::
Linear( const TNatural& n )
  : Superclass( n + 1 )
{
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
ivqML::Model::Regression::Linear< _TReal, _TNatural >::
~Linear( )
{
}

// -------------------------------------------------------------------------
namespace ivqML
{
  namespace Model
  {
    namespace Regression
    {
      template class ivqML_EXPORT Linear< float, unsigned int >;
      template class ivqML_EXPORT Linear< float, unsigned long >;
      template class ivqML_EXPORT Linear< float, unsigned long long >;

      template class ivqML_EXPORT Linear< double, unsigned int >;
      template class ivqML_EXPORT Linear< double, unsigned long >;
      template class ivqML_EXPORT Linear< double, unsigned long long >;

      template class ivqML_EXPORT Linear< long double, unsigned int >;
      template class ivqML_EXPORT Linear< long double, unsigned long >;
      template class ivqML_EXPORT Linear< long double, unsigned long long >;
    } // end namespace
  } // end namespace
} // end namespace

// eof - $RCSfile$
