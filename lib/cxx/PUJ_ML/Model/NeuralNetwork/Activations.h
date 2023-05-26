// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__NeuralNetwork__Activations__h__
#define __PUJ_ML__Model__NeuralNetwork__Activations__h__

#include <string>

namespace PUJ_ML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _M >
      struct Activations
      {
        using TModel = _M;
        using TReal = typename TModel::TReal;
        using TMatrix = typename TModel::TMatrix;
        using TCol = typename TModel::TCol;
        using TRow = typename TModel::TRow;
        using TFunction  = typename TModel::TActivation;

        TFunction operator()( const std::string& n );
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <PUJ_ML/Model/NeuralNetwork/Activations.hxx>

#endif // __PUJ_ML__Model__NeuralNetwork__Activations__h__

// eof - $RCSfile$
