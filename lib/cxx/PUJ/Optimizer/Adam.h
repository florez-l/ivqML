// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Optimizer__Adam__h__
#define __PUJ__Optimizer__Adam__h__

#include <PUJ/Optimizer/Base.h>

namespace PUJ
{
  namespace Optimizer
  {
    /**
     */
    template< class _TModel >
    class Adam
      : public PUJ::Optimizer::Base< _TModel >
    {
    public:
      using TModel     = _TModel;
      using Superclass = PUJ::Optimizer::Base< TModel >;
      using Self       = Adam;

      using TCost   = typename Superclass::TCost;
      using TMatrix = typename Superclass::TMatrix;
      using TScalar = typename Superclass::TScalar;
      using TCol    = typename Superclass::TCol;
      using TRow    = typename Superclass::TRow;
      using TDebug  = typename Superclass::TDebug;

    public:
      Adam( );
      virtual ~Adam( ) = default;

      virtual void AddArguments(
        boost::program_options::options_description* o
        ) override;

      PUJ_ML_Attribute( Alpha, TScalar, 1e-2 );
      PUJ_ML_Attribute( Beta1, TScalar, 0.9 );
      PUJ_ML_Attribute( Beta2, TScalar, 0.999 );

      virtual void Fit( ) override;
    };
  } // end namespace
} // end namespace

#include <PUJ/Optimizer/Adam.hxx>

#endif // __PUJ__Optimizer__Adam__h__

// eof - $RCSfile$
