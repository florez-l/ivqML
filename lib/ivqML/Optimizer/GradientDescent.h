// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__GradientDescent__h__
#define __ivqML__Optimizer__GradientDescent__h__

#include <ivqML/Optimizer/Base.h>

namespace ivqML
{
  namespace Optimizer
  {
    /**
     */
    template< class _TModel >
    class GradientDescent
      : public ivqML::Optimizer::Base< _TModel >
    {
    public:
      using TModel     = _TModel;
      using Self       = GradientDescent;
      using Superclass = ivqML::Optimizer::Base< TModel >;
      using TReal      = typename Superclass::TReal;
      using TNatural   = typename Superclass::TNatural;
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
      GradientDescent( TModel& m );
      virtual ~GradientDescent( ) override;

      const TReal& alpha( ) const;
      void setAlpha( const TReal& a );

      template< class _TX_tr, class _Ty_tr, class _TX_te, class _Ty_te >
      void fit(
        const Eigen::EigenBase< _TX_tr >& bX_train,
        const Eigen::EigenBase< _Ty_tr >& by_train,
        const Eigen::EigenBase< _TX_te >& bX_test,
        const Eigen::EigenBase< _Ty_te >& by_test
        );

    protected:
      TReal m_Alpha { 1e-2 };
    };
  } // end namespace
} // end namespace

#include <ivqML/Optimizer/GradientDescent.hxx>

#endif // __ivqML__Optimizer__GradientDescent__h__

// eof - $RCSfile$
