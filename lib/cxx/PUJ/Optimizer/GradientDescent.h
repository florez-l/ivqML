// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Optimizer__GradientDescent__h__
#define __PUJ__Optimizer__GradientDescent__h__

#include <functional>
#include <PUJ/Traits.h>

namespace PUJ
{
  namespace Optimizer
  {
    /**
     */
    template< class _TScalar, class _TTraits = PUJ::Traits< _TScalar > >
    class GradientDescent
    {
    public:
      PUJ_TraitsMacro( GradientDescent );

      using TCost = std::function< TScalar( const TRow&, TRow* ) >;

      enum InitType
      {
        ZerosInit,
        OnesInit,
        RandomInit
      };

      using TDebug = std::function< void( const TScalar&, const TScalar&, const TRow&, unsigned long long ) >;

    public:
      GradientDescent(
        const TCost& c,
        unsigned long d,
        const Self::InitType& t = Self::RandomInit
        );
      virtual ~GradientDescent( ) = default;

      const TCost& GetCost( ) const;
      const TRow& GetTheta( ) const;

      const TScalar& GetAlpha( ) const;
      const TScalar& GetLambda( ) const;
      const TScalar& GetEpsilon( ) const;
      const unsigned long long& GetMaximumNumberOfIterations( ) const;
      const unsigned long long& GetDebugIterations( ) const;

      void SetAlpha( const TScalar& a );
      void SetLambda( const TScalar& l );
      void SetEpsilon( const TScalar& e );
      void SetMaximumNumberOfIterations( const unsigned long long& i );
      void SetDebugIterations( const unsigned long long& i );
      void SetDebug( TDebug f );

      void Fit( );

    protected:
      TCost m_Cost;
      unsigned long m_Dimensions;

      TScalar m_Alpha;
      TScalar m_Lambda;
      TScalar m_Epsilon;

      unsigned long long m_MaximumNumberOfIterations;
      unsigned long long m_DebugIterations;
      TDebug m_Debug;

      TRow m_Theta;
    };
  } // end namespace
} // end namespace

#endif // __PUJ__Optimizer__GradientDescent__h__

// eof - $RCSfile$
