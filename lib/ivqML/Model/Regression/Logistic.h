// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Regression__Logistic__h__
#define __ivqML__Model__Regression__Logistic__h__

#include <ivqML/Model/Regression/Linear.h>

namespace ivqML
{
  namespace Model
  {
    namespace Regression
    {
      /**
       */
      template< class _S >
      class Logistic
        : public ivqML::Model::Regression::Linear< _S >
      {
      public:
        using Self = Logistic;
        using Superclass = ivqML::Model::Regression::Linear< _S >;

        using TScalar = typename Superclass::TScalar;
        using TNatural = typename Superclass::TNatural;
        using TMatrix = typename Superclass::TMatrix;
        using TMap = typename Superclass::TMap;

      public:
        Logistic( const TNatural& n = 0 );
        virtual ~Logistic( ) = default;

        template< class _X >
        auto evaluate( const Eigen::EigenBase< _X >& iX ) const;

        template< class _X, class _Y >
        void cost(
          TScalar* bG,
          const Eigen::EigenBase< _X >& iX,
          const Eigen::EigenBase< _Y >& iY,
          TScalar* J = nullptr,
          TScalar* buffer = nullptr
          ) const;

        template< class _X >
        auto threshold( const Eigen::EigenBase< _X >& iX ) const;
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Model/Regression/Logistic.hxx>

#endif // __ivqML__Model__Regression__Logistic__h__

// eof - $RCSfile$
