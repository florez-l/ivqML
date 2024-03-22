// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__ActivationFactory__h__
#define __ivqML__Model__NeuralNetwork__ActivationFactory__h__

#include <limits>
#include <string>

namespace ivqML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _M >
      class ActivationFactory
      {
      public:
        using Self        = ActivationFactory;
        using TModel      = _M;
        using TScl        = typename TModel::TScl;
        using TNat        = typename TModel::TNat;
        using TMat        = typename TModel::TMat;
        using TCol        = typename TModel::TCol;
        using TRow        = typename TModel::TRow;
        using TMatMap     = typename TModel::TMatMap;
        using TColMap     = typename TModel::TColMap;
        using TRowMap     = typename TModel::TRowMap;
        using TMatCMap    = typename TModel::TMatCMap;
        using TColCMap    = typename TModel::TColCMap;
        using TRowCMap    = typename TModel::TRowCMap;
        using TActivation = typename TModel::TActivation;
        using TTraits     = std::numeric_limits< TScl >;

      public:
        static TActivation New( const std::string& n );
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Model/NeuralNetwork/ActivationFactory.hxx>

#endif // __ivqML__Model__NeuralNetwork__ActivationFactory__h__

// eof - $RCSfile$
