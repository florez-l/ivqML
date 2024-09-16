// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Common__KMeans__h__
#define __ivqML__Common__KMeans__h__

#include <functional>
#include <string>
#include <ivq/eigen/Config.h>

namespace ivqML
{
  namespace Common
  {
    namespace KMeans
    {
      /**
       */
      template< class _TM, class _TX >
      void RandomInit(
        Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X
        );

      /**
       */
      template< class _TM, class _TX >
      void ForgyInit(
        Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X
        );

      /**
       */
      template< class _TM, class _TX >
      void XXInit(
        Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X
        );

      /**
       */
      template< class _TM, class _TX >
      void Init(
        Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X,
        const std::string& method = "++"
        );

      /**
       */
      template< class _TM, class _TX >
      void Fit(
        Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X,
        std::function<
          bool( const typename _TM::Scalar&, const unsigned long long& )
          >
        debug
        =
        []( const typename _TM::Scalar&, const unsigned long long& )
        ->
        bool
        {
          return( false );
        }
        );

    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Common/KMeans.hxx>

#endif // __ivqML__Common__KMeans__h__

// eof - $RCSfile$
