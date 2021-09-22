// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ/Data/Algorithms.h>
#include <Eigen/Dense>
#include <random>
#include <iostream>
// -------------------------------------------------------------------------
template< class _TMatrix >
void PUJ::Algorithms::
Shuffle( _TMatrix& M, bool columns )
{
  using _N  = unsigned long long;
  using _TP = Eigen::PermutationMatrix< Eigen::Dynamic, Eigen::Dynamic >;

  _N m = ( columns )? M.rows( ): M.cols( );

  std::random_device r;
  std::mt19937 g( r( ) );
  std::uniform_int_distribution< _N > d( 0, m - 1 );

  _TP p( m );
  p.setIdentity( );
  std::random_shuffle(
    p.indices( ).data( ), p.indices( ).data( ) + p.indices( ).size( ),
    [&]( _N i )
    {
      return( d( g ) );
    }
    );
  if( columns )
    M = p * M;
  else
    M = M * p;
}

// -------------------------------------------------------------------------
#include <PUJ_ML_export.h>
#include <PUJ/Traits.h>

#define PUJ_ML_Algorithms_Shuffle( t )                                  \
  template void PUJ_ML_EXPORT                                           \
  PUJ::Algorithms::Shuffle< PUJ::Traits< t >::TMatrix >(                \
    PUJ::Traits< t >::TMatrix&, bool                                    \
    );                                                                  \
  template void PUJ_ML_EXPORT                                           \
  PUJ::Algorithms::Shuffle< PUJ::Traits< t >::TCol >(                   \
    PUJ::Traits< t >::TCol&, bool                                       \
    );                                                                  \
  template void PUJ_ML_EXPORT                                           \
  PUJ::Algorithms::Shuffle< PUJ::Traits< t >::TRow >(                   \
    PUJ::Traits< t >::TRow&, bool                                       \
    )

PUJ_ML_Algorithms_Shuffle( float );
PUJ_ML_Algorithms_Shuffle( double );
PUJ_ML_Algorithms_Shuffle( long double );

// eof - $RCSfile$
