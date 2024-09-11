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

      virtual TReal _E( TBuffer& p ) const
        {
          unsigned int K = this->m_Means.rows( );
          unsigned int F = this->m_Means.cols( );
          unsigned long long KF = K * F;

          TReal* d = p.data( );
          MMatrix W( d, 1, K );
          MMatrix M( d + K, K, F );
          MMatrix C( d + ( K + KF ), KF, F );

          TMatrix EPS = Eigen::Diagonal< TMatrix >::Ones( F, F ) * this->m_EPS;
          // TODO: std::cout << EPS << std::endl;
          // TODO: std::exit( 1 );

          /* TODO
             M -= this->m_Means;
             C += this->m_COVs;
             C.array( ) *= TReal( 0.5 );
          */

          TReal e = 0;
          for( unsigned int k = 0; k < K; ++k )
          {
            TMatrix m1 = this->m_Means.row( k );
            TMatrix m2 = M.row( k );
            TMatrix C1 = this->m_COVs.block( k * F, 0, F, F ).array( ) + this->m_EPS;
            TMatrix C2 = C.block( k * F, 0, F, F ).array( ) + this->m_EPS;
            TReal d1 = C1.determinant( );
            TReal d2 = C2.determinant( );
            TMatrix iC = ( C1 + C2 ) / TReal( 2 );

            TReal le
              =
              (
                ( ( m1 - m2 ) * iC.inverse( ) * ( m1 - m2 ).transpose( ) )
                /
                TReal( 8 )
                )( 0, 0 )
              +
              (
                std::log( iC.determinant( ) / std::sqrt( d1 * d2 ) )
                /
                TReal( 2 )
                );
            e += le * le;

          } // end for
          return( e / TReal( K ) );
        }

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
