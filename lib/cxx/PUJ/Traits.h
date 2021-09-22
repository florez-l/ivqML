// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Traits__h__
#define __PUJ__Traits__h__

#include <Eigen/Core>

// -------------------------------------------------------------------------
#define PUJ_TraitsMacro( _s )                  \
  using Self    = _s;                          \
  using TTraits = _TTraits;                    \
  using TScalar = typename TTraits::TScalar;   \
  using TMatrix = typename TTraits::TMatrix;   \
  using TRow    = typename TTraits::TRow;      \
  using TCol    = typename TTraits::TCol

namespace PUJ
{
  /**
   */
  template< class _TScalar >
  class Traits
  {
  public:
    using TScalar = _TScalar;
    using TMatrix = Eigen::Matrix< TScalar, Eigen::Dynamic, Eigen::Dynamic >;
    using TRow    = Eigen::Matrix< TScalar, 1, Eigen::Dynamic >;
    using TCol    = Eigen::Matrix< TScalar, Eigen::Dynamic, 1 >;
  };

  template< typename Func >
  struct lambda_as_visitor_wrapper
    : Func
  {
    lambda_as_visitor_wrapper( const Func& f )
      : Func( f )
      {
      }

    template< typename S, typename I >
    void init( const S& v, I i, I j )
      {
        return( Func::operator()( v, i, j ) );
      }
  };

  template< typename Mat, typename Func >
  void visit_lambda( const Mat& m, const Func& f )
  {
    lambda_as_visitor_wrapper< Func > visitor( f );
    m.visit( visitor );
  }

} // end namespace

#endif // __PUJ__Traits__h__

// eof - $RCSfile$
