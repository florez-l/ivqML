// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Common__MixtureOfGaussians__h__
#define __ivqML__Common__MixtureOfGaussians__h__

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
    class MixtureOfGaussians
    {
    public:
      using Self  = MixtureOfGaussians;
      using TReal = _TReal;
      using TInt  = short;
      using TBuffer = Eigen::Matrix< TReal, 1, Eigen::Dynamic >;
      using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
      using MMatrix = Eigen::Map< TMatrix >;

      using TDebug = std::function< bool( const TReal& ) >;

    public:
      MixtureOfGaussians( );
      virtual ~MixtureOfGaussians( ) = default;

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
      void label(
        Eigen::EigenBase< _TOutput >& Lb,
        const Eigen::EigenBase< _TInput >& Ib
        );

    protected:
      void _reserve( const TInt& F, const TInt& K );

      template< class _TInput >
      void _C( const _TInput& I );
 
      template< class _TInput >
      _TReal _R( TMatrix& R, const _TInput& I ) const;

      template< class _TOutput >
      void _L( _TOutput& L, const TMatrix& R ) const;

    protected:
      TReal m_EPS { std::numeric_limits< TReal >::epsilon( ) };

      TBuffer m_Parameters;
      MMatrix m_Weights { nullptr, 0, 0 };
      MMatrix m_Means   { nullptr, 0, 0 };
      MMatrix m_COVs    { nullptr, 0, 0 };

      TDebug m_Debug;
    };
  } // end namespace
} // end namespace

#include <ivqML/Common/MixtureOfGaussians.hxx>

#endif // __ivqML__Common__MixtureOfGaussians__h__

// eof - $RCSfile$
