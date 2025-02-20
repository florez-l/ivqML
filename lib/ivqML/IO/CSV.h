// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__IO__CSV__h__
#define __ivqML__IO__CSV__h__

#include <ivqML/Config.h>

namespace ivqML
{
  namespace IO
  {
    /**
     */
    template< class _TD >
    bool ReadCSV(
      Eigen::EigenBase< _TD >& D, const std::string& fname,
      unsigned long long ignore_first_rows = 0,
      const char& separator = ','
      );
  } // end namespace
} // end namespace

#include <ivqML/IO/CSV.hxx>

#endif // __ivqML__IO__CSV__h__

// eof - $RCSfile$
