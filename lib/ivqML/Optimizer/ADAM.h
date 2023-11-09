// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__ADAM__h__
#define __ivqML__Optimizer__ADAM__h__

#include <ivqML/Optimizer/Base.h>

namespace ivqML
{
  namespace Optimizer
  {
    /**
     */
    template< class _C >
    class ADAM
      : public ivqML::Optimizer::Base< _C >
    {
    public:
      using Self = ADAM;
      using Superclass = ivqML::Optimizer::Base< _C >;
      ivqML_Optimizer_Typedefs;

    public:
      ivqMLAttributeMacro( alpha, TScalar, 1e-3 );
      ivqMLAttributeMacro( beta1, TScalar, 0.9 );
      ivqMLAttributeMacro( beta2, TScalar, 0.999 );

    public:
      ADAM( );
      virtual ~ADAM( ) override = default;

      virtual void fit( ) override;
    };
  } // end namespace
} // end namespace

#include <ivqML/Optimizer/ADAM.hxx>

#endif // __ivqML__Optimizer__ADAM__h__

// eof - $RCSfile$
