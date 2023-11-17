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
    template< class _M, class _X = typename _M::TMatrix, class _Y = typename _M::TMatrix >
    class GradientDescent
      : public ivqML::Optimizer::Base< _M, _X, _Y >
    {
    public:
      using Self = GradientDescent;
      using Superclass = ivqML::Optimizer::Base< _M, _X, _Y >;
      ivqML_Optimizer_Typedefs;

    public:
      ivqMLAttributeMacro( alpha, TScalar, 1e-3 );

    public:
      GradientDescent( );
      virtual ~GradientDescent( ) override = default;

      virtual void fit( ) override;
    };
  } // end namespace
} // end namespace

#include <ivqML/Optimizer/GradientDescent.hxx>

#endif // __ivqML__Optimizer__GradientDescent__h__

// eof - $RCSfile$
