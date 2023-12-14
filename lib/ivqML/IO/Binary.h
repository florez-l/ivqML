// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__IO__Binary__h__
#define __ivqML__IO__Binary__h__

#include <ivqML/Config.h>

namespace ivqML
{
  namespace IO
  {
    namespace Binary
    {
      /**
       */
      template< class _M >
      bool Read(
        Eigen::EigenBase< _M >& M,
        const std::string& fname
        );

      /**
       */
      template< class _M >
      bool Write(
        const Eigen::EigenBase< _M >& M,
        const std::string& fname
        );

    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/IO/Binary.hxx>

#endif // __ivqML__IO__Binary__h__

// eof - $RCSfile$
