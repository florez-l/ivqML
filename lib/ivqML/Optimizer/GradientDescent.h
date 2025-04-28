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
      using TModel = _TModel;
      ivqML_TypeTraits( typename TModel::TReal );
      using Self = GradientDescent;
      using Superclass = ivqML::Optimizer::Base< _TModel >;

    protected:
      using TBatchRow = typename Superclass::TBatchRow;
      using TBatch    = typename Superclass::TBatch;
      using TBatches  = typename Superclass::TBatches;
      using TShuffler = typename Superclass::TShuffler;

    public:
      GradientDescent( );
      virtual ~GradientDescent( ) override;

    protected:
      virtual void _fit(
        TModel& model, TBatches& batches, TShuffler shuffler
        ) override;

    protected:
      ivqML_AttributeMacro( learning_rate, Alpha, TReal, TReal( 1e-2 ) );
    };
  } // end namespace
} // end namespace

#include <ivqML/Optimizer/GradientDescent.hxx>

#endif // __ivqML__Optimizer__GradientDescent__h__

// eof - $RCSfile$
