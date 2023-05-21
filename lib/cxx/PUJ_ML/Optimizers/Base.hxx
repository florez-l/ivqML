// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizers__Base__hxx__
#define __PUJ_ML__Optimizers__Base__hxx__

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

// -------------------------------------------------------------------------
template< class _C, class _X, class _Y >
PUJ_ML::Optimizers::Base< _C, _X, _Y >::
Base( TModel& m, const TX& X, const TY& Y )
  : m_Model( &m ),
    m_X( &X ),
    m_Y( &Y )
{
  this->m_Epsilon =
    std::pow(
      TReal( 10 ),
      std::floor(
        std::log10( std::numeric_limits< TReal >::epsilon( ) ) * TReal( 0.5 )
        )
      );

  this->set_debug(
    [](
      const TReal& J,
      const TReal& nG,
      const unsigned long long& epoch
      ) -> bool
    {
      return( false );
    }
    );
}

// -------------------------------------------------------------------------
template< class _C, class _X, class _Y >
void PUJ_ML::Optimizers::Base< _C, _X, _Y >::
set_debug( TDebug d )
{
  this->m_Debug = d;
}

// -------------------------------------------------------------------------
template< class _C, class _X, class _Y >
void PUJ_ML::Optimizers::Base< _C, _X, _Y >::
set_regularization_type_to_ridge( )
{
}

// -------------------------------------------------------------------------
template< class _C, class _X, class _Y >
void PUJ_ML::Optimizers::Base< _C, _X, _Y >::
set_regularization_type_to_LASSO( )
{
}

// -------------------------------------------------------------------------
template< class _C, class _X, class _Y >
void PUJ_ML::Optimizers::Base< _C, _X, _Y >::
_batches( std::vector< std::vector< unsigned long long > >& indices ) const
{
  // Create indices and shuffle them
  unsigned long long m = this->m_X->derived( ).rows( );
  std::vector< unsigned long long > idx( m );
  std::iota( idx.begin( ), idx.end( ), 0 );
  std::random_device rd;
  std::default_random_engine rng( rd( ) );
  std::shuffle( idx.begin( ), idx.end( ), rng );

  // Create groups
  indices.clear( );
  unsigned long long bs = m;
  if( this->m_BatchSize > 0 && this->m_BatchSize < m )
    bs = this->m_BatchSize;
  for( unsigned long long i = 0; i < m; i += bs )
  {
    unsigned long long j = i + bs;
    if( j > m )
      j = m;
    indices.push_back(
      std::vector< unsigned long long >( idx.begin( ) + i, idx.begin( ) + j )
      );
  } // end for
}

#endif // __PUJ_ML__Optimizers__Base__hxx__

// eof - $RCSfile$
