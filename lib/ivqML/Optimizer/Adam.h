// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Adam__h__
#define __ivqML__Optimizer__Adam__h__

#include <ivqML/Optimizer/GradientDescent.h>

namespace ivqML
{
  namespace Optimizer
  {
    /**
     */
    template< class _TModel >
    class Adam
      : public ivqML::Optimizer::GradientDescent< _TModel >
    {
    public:
      using TModel     = _TModel;
      using Self       = Adam;
      using Superclass = ivqML::Optimizer::GradientDescent< TModel >;
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
      Adam( TModel& m );
      virtual ~Adam( ) override;

      const TReal& beta1( ) const;
      const TReal& beta2( ) const;

      void setBeta1( const TReal& b );
      void setBeta2( const TReal& b );

      template< class _TX_tr, class _Ty_tr, class _TX_te, class _Ty_te >
      void fit(
        const Eigen::EigenBase< _TX_tr >& bX_train,
        const Eigen::EigenBase< _Ty_tr >& by_train,
        const Eigen::EigenBase< _TX_te >& bX_test,
        const Eigen::EigenBase< _Ty_te >& by_test
        );

    protected:
      TReal m_Beta1 { 0.9 };
      TReal m_Beta2 { 0.999 };
    };
  } // end namespace
} // end namespace

#include <ivqML/Optimizer/Adam.hxx>

#endif // __ivqML__Optimizer__Adam__h__

// eof - Adam.h
