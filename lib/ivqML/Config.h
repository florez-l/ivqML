// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Config__h__
#define __ivqML__Config__h__

#include <ivqML/Export.h>
#include <Eigen/Core>

// -------------------------------------------------------------------------
#define ivqMLAttributeMacro( _N, _T, _V )        \
  public:                                        \
  const _T _N( ) const                           \
  {                                              \
    return( this->m_##_N );                      \
  }                                              \
  void set_##_N( const _T& v )                   \
  {                                              \
    this->m_##_N = v;                            \
  }                                              \
  protected:                                     \
  _T m_##_N { _T( _V ) }

#endif // __ivqML__Config__h__

// eof - $RCSfile$
