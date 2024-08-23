// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Common__PCA__h__
#define __ivqML__Common__PCA__h__

#include <tuple>
#include <ivq/eigen/Config.h>

namespace ivqML
{
  namespace Common
  {
    /**
     */
    template< class _TData, class _TReal = float >
    auto EigenAnalysis( const Eigen::EigenBase< _TData >& X );

    /**
     */
    template< class _TData, class _TMean, class _TMatrix, class _TValues, class _TReal = float >
    auto PCA(
      const Eigen::EigenBase< _TData >& X,
      const std::tuple< _TMean, _TMatrix, _TValues >& E,
      const _TReal& p = 1
      );

    /**
     */
    template< class _TData, class _TReal = float >
    auto PCA( const Eigen::EigenBase< _TData >& X, const _TReal& p = 1 );

  } // end namespace
} // end namespace

#include <ivqML/Common/PCA.hxx>

#endif // __ivqML__Common__PCA__h__

// eof - $RCSfile$
