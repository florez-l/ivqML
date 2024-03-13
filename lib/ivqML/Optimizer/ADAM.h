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
    template< class _TCost >
    class ADAM
      : public ivqML::Optimizer::Base< _TCost >
    {
    public:
      using Self       = ADAM;
      using Superclass = ivqML::Optimizer::Base< _TCost >;
      using TCost      = typename Superclass::TCost;
      using TModel     = typename Superclass::TModel;
      using TScalar    = typename Superclass::TScalar;
      using TNatural   = typename Superclass::TNatural;
      using TMat       = typename Superclass::TMat;
      using TCol       = typename Superclass::TCol;
      using TRow       = typename Superclass::TRow;
      using TMatMap    = typename Superclass::TMatMap;
      using TColMap    = typename Superclass::TColMap;
      using TRowMap    = typename Superclass::TRowMap;
      using TMatCMap   = typename Superclass::TMatCMap;
      using TColCMap   = typename Superclass::TColCMap;
      using TRowCMap   = typename Superclass::TRowCMap;

    public:
      ivqMLAttributeMacro( alpha, TScalar, 1e-4 );
      ivqMLAttributeMacro( beta1, TScalar, 0.9 );
      ivqMLAttributeMacro( beta2, TScalar, 0.999 );

    public:
      ADAM( );
      virtual ~ADAM( ) = default;

      virtual void fit( ) override;

    private:
      ADAM( const Self& ) = delete;
      Self& operator=( const Self& ) = delete;
    };
  } // end namespace
} // end namespace

#include <ivqML/Optimizer/ADAM.hxx>

#endif // __ivqML__Optimizer__ADAM__h__

// eof - $RCSfile$
