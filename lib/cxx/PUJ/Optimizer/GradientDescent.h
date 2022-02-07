// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizer__GradientDescent__h__
#define __PUJ_ML__Optimizer__GradientDescent__h__

#include <functional>

namespace PUJ_ML
{
  namespace Optimizer
  {
    /**
     */
    template< class _M >
    class GradientDescent
    {
    public:
      using Self       = GradientDescent;
      using TModel     = _M;
      using TScalar    = typename _M::TScalar;
      using TMatrix    = typename _M::TMatrix;
      using TRow       = typename _M::TRow;
      using TCol       = typename _M::TCol;
      using TCost      = typename _M::Cost;

      using TDebug =
        std::function< bool( unsigned long long, TScalar, bool ) >;

    public:
      enum ERegularization
      {
        RidgeReg = 0,
        LASSOReg
      };

    public:
      GradientDescent( );
      virtual ~GradientDescent( ) = default;

      TCost* GetCost( ) const;
      void SetCost( TCost& c );

      const TScalar& GetLearningRate( ) const;
      void SetLearningRate( const TScalar& a );

      const TScalar& GetRegularizationCoefficient( ) const;
      void SetRegularizationCoefficient( const TScalar& l );

      const ERegularization& GetRegularization( ) const;
      void SetRegularizationToRidge( );
      void SetRegularizationToLASSO( );

      const TScalar& GetEpsilon( ) const;
      void SetEpsilon( const TScalar& e );

      const unsigned long long& GetNumberOfEpochs( ) const;
      void SetNumberOfEpochs( const unsigned long long& n );

      const unsigned long long& GetDebugStep( ) const;
      void SetDebugStep( const unsigned long long& d );

      void UnsetDebug( );
      void SetDebug( TDebug d );

      void Fit( );

    protected:
      TCost*  m_Cost { nullptr };

      TScalar m_A { 0.01 };
      TScalar m_L { 0 };
      ERegularization m_LType;
      unsigned long long m_N { 1000 };
      unsigned long long m_D { 100 };

      TDebug m_Debug;

      TScalar m_E;
    };
  } // end namespace
} // end namespace

#include <PUJ_ML/Optimizer/GradientDescent.hxx>

#endif // __PUJ_ML__Optimizer__GradientDescent__h__

// eof - $RCSfile$
