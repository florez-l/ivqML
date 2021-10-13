// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Optimizer__GradientDescent__h__
#define __PUJ__Optimizer__GradientDescent__h__

#include <functional>

namespace PUJ
{
  namespace Optimizer
  {
    /**
     */
    template< class _TModel >
    class GradientDescent
    {
    public:
      using TModel = _TModel;
      using Self = GradientDescent;

      using TCost   = typename TModel::Cost;
      using TMatrix = typename TModel::TMatrix;
      using TScalar = typename TModel::TScalar;
      using TCol    = typename TModel::TCol;
      using TRow    = typename TModel::TRow;
      
      enum RegularizationType
      {
        RidgeRegType = 0,
        LassoRegType
      };

      using TDebug =
        std::function< bool( unsigned long long, TScalar, bool ) >;

    public:
      GradientDescent( TCost* cost );
      virtual ~GradientDescent( ) = default;

      PUJ_ML_Attribute( Cost, TCost*, nullptr );
      PUJ_ML_Attribute( Alpha, TScalar, 1e-2 );
      PUJ_ML_Attribute( Lambda, TScalar, 0 );
      PUJ_ML_Attribute( Epsilon, TScalar, 1e-8 );
      PUJ_ML_Attribute( MaximumNumberOfIterations, unsigned long long, 2000 );
      PUJ_ML_Attribute( DebugIterations, unsigned long long, 100 );

      PUJ_ML_Attribute( RegularizationType, RegularizationType, RidgeRegType );
      void SetRegularizationTypeToRidge( );
      void SetRegularizationTypeToLasso( );

      const unsigned long long& GetRealIterations( ) const;
      void SetDebug( TDebug d );

      virtual void Fit( );

    protected:
      void _Regularize( TScalar& J, TRow& g );

    protected:
      unsigned long long m_RealIterations = { 0 };
      TDebug m_Debug;
    };
  } // end namespace
} // end namespace

#include <PUJ/Optimizer/GradientDescent.hxx>

#endif // __PUJ__Optimizer__GradientDescent__h__

// eof - $RCSfile$
