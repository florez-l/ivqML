// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Common__ImageHelpers__h__
#define __ivqML__Common__ImageHelpers__h__

namespace ivqML
{
  namespace Common
  {
    namespace ImageHelpers
    {
      /**
       */
      template< class _I, class _M >
      _M meshgrid( const _I* image );

    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Common/ImageHelpers.hxx>

#endif // __ivqML__Common__ImageHelpers__h__

// eof - $RCSfile$
