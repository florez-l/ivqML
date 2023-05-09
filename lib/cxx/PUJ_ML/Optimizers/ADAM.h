// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizers__ADAM__h__
#define __PUJ_ML__Optimizers__ADAM__h__

#include <PUJ_ML/Optimizers/Base.h>

namespace PUJ_ML
{
  namespace Optimizers
  {
    /**
     */
    template< class _C, class _X, class _Y >
    class ADAM
      : public PUJ_ML::Optimizers::Base< _C, _X, _Y >
    {
    public:
      using Self = ADAM;
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

    public:
      PUJ_ML_Attribute( TReal, Beta1, beta1, 0.9 );
      PUJ_ML_Attribute( TReal, Beta2, beta2, 0.999 );

    public:
      ADAM( TModel& m, const TX& X, const TY& Y );
      virtual ~ADAM( ) = default;

      virtual void fit( ) override;

    protected:
      TReal m_SlideEpsilon;
    };
  } // end namespace
} // end namespace

#include <PUJ_ML/Optimizers/ADAM.hxx>

#endif // __PUJ_ML__Optimizers__ADAM__h__

// eof - $RCSfile$
