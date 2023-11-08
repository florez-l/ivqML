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
    template< class _C >
    class GradientDescent
      : public ivqML::Optimizer::Base< _C >
    {
    public:
      using Self = GradientDescent;
      using Superclass = ivqML::Optimizer::Base< _C >;
      using TCost = typename Superclass::TCost;
      using TModel = typename Superclass::TModel;
      using TDX = typename Superclass::TDX;
      using TDY = typename Superclass::TDY;
      using TX = typename Superclass::TX;
      using TY = typename Superclass::TY;
      using TScalar = typename Superclass::TScalar;
      using TNatural = typename Superclass::TNatural;
      using TMatrix = typename Superclass::TMatrix;
      using TMap = typename Superclass::TMap;
      using TConstMap = typename Superclass::TConstMap;
      using TResult = typename Superclass::TResult;

    public:
      ivqMLAttributeMacro( alpha, TScalar, 1e-3 );

    public:
      GradientDescent( TModel& m, const TX& iX, const TY& iY );
      virtual ~GradientDescent( ) override = default;

      virtual void fit( ) override;
    };
  } // end namespace
} // end namespace

#include <ivqML/Optimizer/GradientDescent.hxx>

#endif // __ivqML__Optimizer__GradientDescent__h__

// eof - $RCSfile$
