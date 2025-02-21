// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Helpers__SplitDatasetWithBinaryLabeling__h__
#define __ivqML__Helpers__SplitDatasetWithBinaryLabeling__h__

#include <ivqML/Config.h>
#include <vector>

namespace ivqML
{
  namespace Helpers
  {
    /**
     */
    template< class _TR >
    struct SplitDatasetWithBinaryLabeling
    {
      using _TIdx = Eigen::Index;

      void init( const _TR& v, const _TIdx& r, const _TIdx& c );
      void operator()( const _TR& v, const _TIdx& r, const _TIdx& c );
      void finish( const _TR& s );

      std::vector< _TIdx > Z, O, Tr, Te;
    };
  } // end namespace
} // end namespace

#endif // __ivqML__Helpers__SplitDatasetWithBinaryLabeling__h__

// eof - $RCSfile$
