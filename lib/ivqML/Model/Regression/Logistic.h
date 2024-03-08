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
      template< class _TScalar >
      class Logistic
        : public ivqML::Model::Regression::Linear< _TScalar >
      {
      public:
      public:
        using Self       = Logistic;
        using Superclass = ivqML::Model::Regression::Linear< _TScalar >;
        using TScalar    = typename Superclass::TScalar;
        using TNatural   = typename Superclass::TNatural;
        using TMatrix    = typename Superclass::TMatrix;
        using TColumn    = typename Superclass::TColumn;
        using TRow       = typename Superclass::TRow;

      public:
        Logistic( const TNatural& n = 0 );
        virtual ~Logistic( ) = default;

        template< class _TInputX >
        auto evaluate(
          const Eigen::EigenBase< _TInputX >& iX, TScalar* iB = nullptr
          ) const;

        /* TODO
           template< class _X, class _Y >
           void cost(
           TScalar* bG,
           const Eigen::EigenBase< _X >& iX,
           const Eigen::EigenBase< _Y >& iY,
           TScalar* J = nullptr,
           TScalar* buffer = nullptr
           ) const;
        */

        template< class _TInputX >
        auto threshold( const Eigen::EigenBase< _TInputX >& iX ) const;

      private:
        Logistic( const Self& other ) = delete;
        Self& operator=( const Self& other ) = delete;
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Model/Regression/Logistic.hxx>

#endif // __ivqML__Model__Regression__Logistic__h__

// eof - $RCSfile$
