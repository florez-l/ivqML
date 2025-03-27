// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__IO__h__
#define __ivqML__IO__h__

#include <ivqML/Config.h>

namespace ivqML
{
  namespace IO
  {
    /**
     */
    template< class _TReal >
    void ReadIDX(
      Eigen::Matrix< _TReal, Eigen::Dynamic, Eigen::Dynamic >& D,
      const std::string& fname
      );

    /**
     */
    template< class _TReal >
    void ReadMNIST(
      Eigen::Matrix< _TReal, Eigen::Dynamic, Eigen::Dynamic >& Xtr,
      Eigen::Matrix< _TReal, Eigen::Dynamic, Eigen::Dynamic >& Ytr,
      Eigen::Matrix< _TReal, Eigen::Dynamic, Eigen::Dynamic >& Xte,
      Eigen::Matrix< _TReal, Eigen::Dynamic, Eigen::Dynamic >& Yte,
      const std::string& dname
      );

  } // end namespace
} // end namespace

#endif // __ivqML__IO__IO__h__

// eof - $RCSfile$
