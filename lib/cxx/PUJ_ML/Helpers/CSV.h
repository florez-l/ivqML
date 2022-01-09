// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Helpers__CSV__h__
#define __PUJ__Helpers__CSV__h__

#include <string>
#include <PUJ_ML/Export.h>

namespace PUJ_ML
{
  namespace Helpers
  {
    namespace CSV
    {
      /**
       */
      template< class _TMatrix >
      _TMatrix Read(
        const std::string& fname,
        bool ignore_first_row = false,
        const std::string& separator = ", ;\t"
        );

      /**
       */
      template< class _TMatrix >
      bool Write(
        const _TMatrix& data,
        const std::string& fname,
        const char& separator = ','
        );

    } // end namespace
  } // end namespace
} // end namespace

#endif // __PUJ__Helpers__CSV__h__

// eof - $RCSfile$
