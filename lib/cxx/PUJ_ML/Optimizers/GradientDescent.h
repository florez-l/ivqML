// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizers__GradientDescent__h__
#define __PUJ_ML__Optimizers__GradientDescent__h__

#include <PUJ_ML/Optimizers/Base.h>

namespace PUJ_ML
{
  namespace Optimizers
  {
    /**
     */
    template< class _C, class _X, class _Y >
    class GradientDescent
      : public PUJ_ML::Optimizers::Base< _C, _X, _Y >
    {
    public:
      using Self = GradientDescent;
      using Superclass = PUJ_ML::Optimizers::Base< _C, _X, _Y >;

      using TCost = typename Superclass::TCost;
      using TX = typename Superclass::TX;
      using TY = typename Superclass::TY;
      using TModel = typename Superclass::TModel;
      using TReal = typename Superclass::TReal;
      using TMatrix = typename Superclass::TMatrix;
      using TCol = typename Superclass::TCol;
      using TRow = typename Superclass::TRow;
      using MMatrix = typename Superclass::MMatrix;
      using MCol = typename Superclass::MCol;
      using MRow = typename Superclass::MRow;
      using ConstMMatrix = typename Superclass::ConstMMatrix;
      using ConstMCol = typename Superclass::ConstMCol;
      using ConstMRow = typename Superclass::ConstMRow;

    public:
      GradientDescent( TModel& m, const TX& X, const TY& Y );
      virtual ~GradientDescent( ) = default;

      virtual void fit( ) override;
    };
  } // end namespace
} // end namespace

#include <PUJ_ML/Optimizers/GradientDescent.hxx>

#endif // __PUJ_ML__Optimizers__GradientDescent__h__

// eof - $RCSfile$
