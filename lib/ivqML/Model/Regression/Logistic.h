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
        using TMat       = typename Superclass::TMat;
        using TCol       = typename Superclass::TCol;
        using TRow       = typename Superclass::TRow;
        using TMatMap    = typename Superclass::TMatMap;
        using TColMap    = typename Superclass::TColMap;
        using TRowMap    = typename Superclass::TRowMap;
        using TMatCMap   = typename Superclass::TMatCMap;
        using TColCMap   = typename Superclass::TColCMap;
        using TRowCMap   = typename Superclass::TRowCMap;

      public:
        Logistic( const TNatural& n = 0 );
        virtual ~Logistic( ) = default;

        template< class _TInputX >
        auto eval( const Eigen::EigenBase< _TInputX >& iX ) const;

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
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Model/Regression/Logistic.hxx>

#endif // __ivqML__Model__Regression__Logistic__h__

// eof - $RCSfile$
