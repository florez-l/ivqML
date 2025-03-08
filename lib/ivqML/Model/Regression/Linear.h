// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Regression__Linear__h__
#define __ivqML__Model__Regression__Linear__h__

#include <ivqML/Model/Base.h>

namespace ivqML
{
  namespace Model
  {
    namespace Regression
    {
      /**
       */
      template< class _TReal, class _TNatural = unsigned long long >
      class Linear
        : public ivqML::Model::Base< _TReal, _TNatural >
      {
      public:
        using TReal      = _TReal;
        using TNatural   = _TNatural;
        using Self       = Linear;
        using Superclass = ivqML::Model::Base< TReal, TNatural >;
        using TMat       = typename Superclass::TMat;
        using TCol       = typename Superclass::TCol;
        using TRow       = typename Superclass::TRow;
        using TMatMap    = typename Superclass::TMatMap;
        using TCMatMap   = typename Superclass::TCMatMap;
        using TColMap    = typename Superclass::TColMap;
        using TCColMap   = typename Superclass::TCColMap;
        using TRowMap    = typename Superclass::TRowMap;
        using TCRowMap   = typename Superclass::TCRowMap;

      public:
        Linear( const TNatural& n = 1 );
        virtual ~Linear( ) override;

        TReal& operator[]( const TNatural& i );
        const TReal& operator[]( const TNatural& i ) const;

        template< class _TX >
        auto operator()( const Eigen::EigenBase< _TX >& X ) const;

        /**
         * TODO: Use of L1 regularization is not yet solved
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
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Model/Regression/Linear.hxx>

#endif // __ivqML__Model__Regression__Linear__h__

// eof - $RCSfile$
