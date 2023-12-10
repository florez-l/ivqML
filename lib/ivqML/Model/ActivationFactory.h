// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__ActivationFactory__h__
#define __ivqML__Model__ActivationFactory__h__

#include <limits>
#include <string>

namespace ivqML
{
  namespace Model
  {
    /**
     */
    template< class _M >
    class ActivationFactory
    {
    public:
      using Self        = ActivationFactory;
      using TModel      = _M;
      using TScalar     = typename TModel::TScalar;
      using TMap        = typename TModel::TMap;
      using TMatrix     = typename TModel::TMatrix;
      using TActivation = typename TModel::TActivation;
      using TTraits     = std::numeric_limits< TScalar >;

    public:
      static TActivation New( const std::string& n );
    };
  } // end namespace
} // end namespace

#include <ivqML/Model/ActivationFactory.hxx>

#endif // __ivqML__Model__ActivationFactory__h__

// eof - $RCSfile$
