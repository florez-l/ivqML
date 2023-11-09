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
