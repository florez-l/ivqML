// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__IO__CSV__h__
#define __PUJ_ML__IO__CSV__h__

#include <string>
#include <Eigen/Core>

namespace PUJ_ML
{
  namespace IO
  {
    namespace CSV
    {
      /**
       */
      template< class _X >
      bool read(
        Eigen::EigenBase< _X >& X, const std::string& fname,
        unsigned int ignored_rows = 0,
        const char& separator = ','
        );

    } // end namespace
  } // end namespace
} // end namespace

#include <PUJ_ML/IO/CSV.hxx>

#endif // __PUJ_ML__IO__CSV__h__

// eof - $RCSfile$
