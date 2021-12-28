// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Optimizer__Base__h__
#define __PUJ__Optimizer__Base__h__

#include <functional>
#include <boost/program_options.hpp>

namespace PUJ
{
  namespace Optimizer
  {
    /**
     */
    template< class _TModel >
    class Base
    {
    public:
      using TModel = _TModel;
      using Self = Base;

      using TCost   = typename TModel::Cost;
      using TMatrix = typename TModel::TMatrix;
      using TScalar = typename TModel::TScalar;
      using TCol    = typename TModel::TCol;
      using TRow    = typename TModel::TRow;

      enum RegularizationType
      {
        RidgeRegType = 0,
        LASSORegType
      };

      using TDebug =
        std::function< bool( unsigned long long, TScalar, TScalar, bool ) >;

    public:
      Base( );
      virtual ~Base( ) = default;

      void SetCost( TCost* cost );
      virtual void AddArguments(
        boost::program_options::options_description* o
        );

      PUJ_ML_Attribute( Cost, TCost*, nullptr );
      PUJ_ML_Attribute( Lambda, TScalar, 0 );
      PUJ_ML_Attribute(
        Epsilon, TScalar, std::numeric_limits< TScalar >::epsilon( )
        );
      PUJ_ML_Attribute( MaximumNumberOfIterations, unsigned long long, 1000 );
      PUJ_ML_Attribute( DebugIterations, unsigned long long, 100 );

      PUJ_ML_Attribute( RegularizationType, RegularizationType, RidgeRegType );
      void SetRegularizationTypeToRidge( );
      void SetRegularizationTypeToLASSO( );

      const unsigned long long& GetIterations( ) const;
      void SetDebug( TDebug d );

      virtual void Fit( ) = 0;

    protected:
      void _Regularize( TScalar& J, TRow& g );

    protected:
      unsigned long long m_Iterations = { 0 };
      TDebug m_Debug;
    };
  } // end namespace
} // end namespace

#include <PUJ/Optimizer/Base.hxx>

#endif // __PUJ__Optimizer__Base__h__

// eof - $RCSfile$
