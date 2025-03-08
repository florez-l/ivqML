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
      template< class _TReal, class _TNatural = unsigned long long >
      class Logistic
        : public ivqML::Model::Regression::Linear< _TReal, _TNatural >
      {
      public:
        using TReal      = _TReal;
        using TNatural   = _TNatural;
        using Superclass
        =
          ivqML::Model::Regression::Linear< TReal, TNatural >;
        using Self       = Logistic;
        using TMat       = typename Superclass::TMat;
        using TCol       = typename Superclass::TCol;
        using TRow       = typename Superclass::TRow;
        using TMatMap    = typename Superclass::TMatMap;
        using TCMatMap   = typename Superclass::TCMatMap;
        using TColMap    = typename Superclass::TColMap;
        using TCColMap   = typename Superclass::TCColMap;
        using TRowMap    = typename Superclass::TRowMap;
        using TCRowMap   = typename Superclass::TCRowMap;

      protected:
        using TIdx = Eigen::Index;

      public:
        Logistic( const TNatural& n = 1 );
        virtual ~Logistic( ) override;

        template< class _TX >
        auto operator()(
          const Eigen::EigenBase< _TX >& X, const bool& threshold = false
          ) const;

        /**
         * TODO: This method has no sense in a logistic regression
         */
        template< class _TX, class _Ty >
        void fit(
          const Eigen::EigenBase< _TX >& bX,
          const Eigen::EigenBase< _Ty >& by,
          const TReal& L1 = 0, const TReal& L2 = 0
          );

        template< class _TG, class _TX, class _Ty >
        TReal cost_gradient(
          Eigen::EigenBase< _TG >& G,
          const Eigen::EigenBase< _TX >& bX,
          const Eigen::EigenBase< _Ty >& by,
          const TReal& L1, const TReal& L2
          );

        template< class _TX, class _Ty >
        TReal cost(
          const Eigen::EigenBase< _TX >& X,
          const Eigen::EigenBase< _Ty >& y
          );

      protected:
        /**
         */
        struct SVisitor
        {
          SVisitor( const TCol& Z );
          void init( const TReal& y, const TIdx& i, const TIdx& j );
          void operator()( const TReal& y, const TIdx& i, const TIdx& j );

          const TCol* Z;
          TReal J;
          TReal E;
          TReal D;
        };
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Model/Regression/Logistic.hxx>

#endif // __ivqML__Model__Regression__Logistic__h__

// eof - $RCSfile$
