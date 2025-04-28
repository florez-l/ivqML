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
      using TModel = _TModel;
      ivqML_TypeTraits( typename TModel::TReal );
      using Self = Adam;
      using Superclass = ivqML::Optimizer::GradientDescent< _TModel >;

    protected:
      using TBatchRow = typename Superclass::TBatchRow;
      using TBatch    = typename Superclass::TBatch;
      using TBatches  = typename Superclass::TBatches;
      using TShuffler = typename Superclass::TShuffler;

    public:
      Adam( );
      virtual ~Adam( ) override;

    protected:
      virtual void _fit(
        TModel& model, TBatches& batches, TShuffler shuffler
        ) override;

    protected:
      ivqML_AttributeMacro( beta1, Beta1, TReal, TReal( 0.9 ) );
      ivqML_AttributeMacro( beta2, Beta2, TReal, TReal( 0.999 ) );
    };
  } // end namespace
} // end namespace

#include <ivqML/Optimizer/Adam.hxx>

#endif // __ivqML__Optimizer__Adam__h__

// eof - $RCSfile$
