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
      template< class _TScl >
      class Logistic
        : public ivqML::Model::Regression::Linear< _TScl >
      {
      public:
      public:
        using Self       = Logistic;
        using Superclass = ivqML::Model::Regression::Linear< _TScl >;
        using TScl       = typename Superclass::TScl;
        using TNat       = typename Superclass::TNat;
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
        Logistic( const TNat& n = 0 );
        virtual ~Logistic( ) = default;

        template< class _TInputX >
        auto eval( const Eigen::EigenBase< _TInputX >& iX ) const;

        /* TODO
           template< class _X, class _Y >
           void cost(
           TScl* bG,
           const Eigen::EigenBase< _X >& iX,
           const Eigen::EigenBase< _Y >& iY,
           TScl* J = nullptr,
           TScl* buffer = nullptr
           ) const;
        */

        template< class _TInputX >
        auto threshold( const Eigen::EigenBase< _TInputX >& iX ) const;

      private:
        Logistic( const Self& ) = delete;
        Self& operator=( const Self& ) = delete;
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Model/Regression/Logistic.hxx>

#endif // __ivqML__Model__Regression__Logistic__h__

// eof - $RCSfile$
