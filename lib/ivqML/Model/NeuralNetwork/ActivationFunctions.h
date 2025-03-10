// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__ActivationFunctions__h__
#define __ivqML__Model__NeuralNetwork__ActivationFunctions__h__

#include <functional>
#include <ivqML/Config.h>

namespace ivqML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _TReal >
      class ActivationFunctions
      {
      public:
        using TReal = _TReal;
        using Self  = ActivationFunctions;
        using TMat  = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
        using TMap  = Eigen::Map< TMat >;

        using TUnary = std::function< TReal( const TReal&, bool ) >;
        using TFunction = std::function< void( TMap&, const TMap&, bool ) >;

      public:
        static TFunction Get( const std::string& a );
      };
    } // end namespace
  } // end namespace
} // end namespace

#endif // __ivqML__Model__NeuralNetwork__ActivationFunctions__h__

// eof - $RCSfile$
