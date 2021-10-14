// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Data__CSV__h__
#define __PUJ__Data__CSV__h__

#include <string>

namespace PUJ
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

  } // end namespace
} // end namespace

#include <PUJ/Data/CSV.hxx>

#endif // __PUJ__Data__CSV__h__

// eof - $RCSfile$
