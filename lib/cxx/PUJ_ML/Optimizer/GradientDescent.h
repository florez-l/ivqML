// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizer__GradientDescent__h__
#define __PUJ_ML__Optimizer__GradientDescent__h__

#include <limits>

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

    public:
      GradientDescent( )
      {
      }

      virtual ~GradientDescent( ) = default;
      
      TCost* GetCost( ) const
      {
        return( this->m_Cost );
      }
      void SetCost( TCost* c ) const
      {
        this->m_Cost = c;
      }
      
      const TScalar& GetLearningRate( ) const
      {
        return( this->m_A );
      }
      void SetLearningRate( const TScalar& a )
      {
        this->m_A = a;
      }

      const TScalar& GetEpsilon( ) const
      {
        return( this->m_E );
      }
      void SetEpsilon( const TScalar& e )
      {
        this->m_E = e;
      }

      const unsigned long long& GetMaxNumberOfIterations( ) const
      {
        return( this->m_N );
      }
      void SetMaxNumberOfIterations( const unsigned long long& n )
      {
        this->m_N = n;
      }

      const unsigned long long& GetNumberOfDebugIterations( ) const
      {
        return( this->m_D );
      }
      void GetNumberOfDebugIterations( const unsigned long long& d )
      {
        this->m_D = d;
      }
      
      void Fit( )
      {
      }
      
    protected:
      TCost*  m_Cost { nullptr };
      
      TScalar m_A { 0.01 };
      TScalar m_E { std::numeric_limits< TScalar >::epsilon };
      unsigned long long m_N { 1000 };
      unsigned long long m_D { 100 };
    };
  } // end namespace
} // end namespace

#endif // __PUJ_ML__Optimizer__GradientDescent__h__

// eof - $RCSfile$
