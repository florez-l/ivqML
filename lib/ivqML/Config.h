// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Config__h__
#define __ivqML__Config__h__



#include <iostream>




#include <ivqML/Export.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <ivq/eigen/Config.h>

// -------------------------------------------------------------------------
#define ivqML_TypeTraits( _real )                                       \
  using TReal    = _real;                                               \
  using TNatural = unsigned long long;                                  \
  using TColumn  = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;           \
  using TRow     = Eigen::Matrix< TReal, 1, Eigen::Dynamic >;           \
  using TMatrix  = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >

// -------------------------------------------------------------------------
#define ivqML_AttributeMacro( _g, _n, _t, _d )  \
  public:                                       \
  virtual const _t& _g( ) const                 \
  {                                             \
    return( this->m_##_n );                     \
  }                                             \
  void set_##_g( const _t& v )                  \
  {                                             \
    this->m_##_n = v;                           \
  }                                             \
  protected:                                    \
  _t m_##_n { _d }

#endif // __ivqML__Config__Config__h__

// eof - $RCSfile$
