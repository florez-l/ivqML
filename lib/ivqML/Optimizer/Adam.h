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

      using TNatural  = typename Superclass::TNatural;
      using TReal     = typename Superclass::TReal;
      using TMatrix   = typename Superclass::TMatrix;
      using TRow      = typename Superclass::TRow;
      using TMap      = typename Superclass::TMap;
      using TBatch    = typename Superclass::TBatch;
      using TBatches  = typename Superclass::TBatches;
      using TDebugger = typename Superclass::TDebugger;

    public:
      Adam(
        const TReal* Xtr, const TReal* Ytr,
        const TNatural& Mtr
        );
      Adam(
        const TReal* Xtr, const TReal* Ytr,
        const TReal* Xte, const TReal* Yte,
        const TNatural& Mtr, const TNatural& Mte
        );
      virtual ~Adam( );

      const TReal& beta1( ) const;
      void set_beta1( const TReal& b );

      const TReal& beta2( ) const;
      void set_beta2( const TReal& b );

    protected:
      virtual void _fit( TModel* model, const TBatches& batches ) override;

    protected:
      TReal m_Beta1 { 0.9   };
      TReal m_Beta2 { 0.999 };
    };

  } // end namespace
} // end namespace

#include <ivqML/Optimizer/Adam.hxx>

#endif // __ivqML__Optimizer__Adam__h__

// eof - $RCSfile$
