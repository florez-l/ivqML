// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Common__KMeans__h__
#define __ivqML__Common__KMeans__h__

#include <functional>
#include <limits>
#include <ivq/eigen/Config.h>

namespace ivqML
{
  namespace Common
  {
    /**
     */
    template< class _TReal >
    class KMeans
    {
    public:
      using Self  = KMeans;
      using TReal = _TReal;
      using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;

      using TDebug = std::function< bool( const TReal& ) >;

    public:
      KMeans( );
      virtual ~KMeans( ) = default;

      void set_debug( TDebug d );

      template< class _TInput >
      void init_random(
        const Eigen::EigenBase< _TInput >& Ib,
        const unsigned long long& K
        );

      template< class _TInput >
      void init_XX(
        const Eigen::EigenBase< _TInput >& Ib,
        const unsigned long long& K
        );

      template< class _TInput >
      void init_Forgy(
        const Eigen::EigenBase< _TInput >& Ib,
        const unsigned long long& K
        );

      template< class _TInput >
      void fit( const Eigen::EigenBase< _TInput >& Ib );

      template< class _TOutput, class _TInput >
      auto label(
        Eigen::EigenBase< _TOutput >& Lb,
        const Eigen::EigenBase< _TInput >& Ib
        );

    protected:
      TReal m_EPS { std::numeric_limits< TReal >::epsilon( ) };

      TMatrix        m_Means[ 2 ];
      unsigned short m_ActiveMean;

      TDebug m_Debug;
    };
  } // end namespace
} // end namespace

#include <ivqML/Common/KMeans.hxx>

#endif // __ivqML__Common__KMeans__h__

// eof - $RCSfile$
