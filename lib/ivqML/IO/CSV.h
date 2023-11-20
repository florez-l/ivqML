// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__IO__CSV__h__
#define __ivqML__IO__CSV__h__

namespace ivqML
{
  namespace IO
  {
    namespace CSV
    {
      /**
       */
      template< class _M >
      bool Read(
        Eigen::EigenBase< _M >& M,
        const std::string& fname,
        unsigned long long ignore_first_rows = 0,
        const char& separator = ','
        );

      /**
       */
      template< class _M >
      bool Write(
        const Eigen::EigenBase< _M >& M,
        const std::string& fname,
        const char& separator = ','
        );

    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/IO/CSV.hxx>

#endif // __ivqML__IO__CSV__h__

// eof - $RCSfile$
