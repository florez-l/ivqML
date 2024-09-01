// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Common__KMeans__hxx__
#define __ivqML__Common__KMeans__hxx__

#include <algorithm>
#include <numeric>
#include <random>

// -------------------------------------------------------------------------
template< class _TReal >
ivqML::Common::KMeans< _TReal >::
KMeans( )
{
  this->m_Debug = []( const TReal& mse ) -> bool { return( false ); };
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Common::KMeans< _TReal >::
set_debug( TDebug d )
{
  this->m_Debug = d;
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TInput >
void ivqML::Common::KMeans< _TReal >::
init_random(
  const Eigen::EigenBase< _TInput >& Ib, const unsigned long long& K
  )
{
  auto I = Ib.derived( ).template cast< TReal >( );
  std::vector< unsigned long long > idx( I.cols( ) );
  std::iota( idx.begin( ), idx.end( ), 0 );
  std::shuffle(
    idx.begin( ), idx.end( ),
    std::default_random_engine(
      std::chrono::system_clock::now( ).time_since_epoch( ).count( )
      )
    );

  std::vector< unsigned long long > means( idx.data( ), idx.data( ) + K );
  this->m_Means[ 0 ] = I( means, ivq_EIGEN_ALL );
  this->m_Means[ 1 ]
    =
    TMatrix::Zero( this->m_Means[ 0 ].rows( ), this->m_Means[ 0 ].cols( ) );
  this->m_ActiveMean = 0;
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TInput >
void ivqML::Common::KMeans< _TReal >::
init_XX(
  const Eigen::EigenBase< _TInput >& Ib,
  const unsigned long long& K
  )
{
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TInput >
void ivqML::Common::KMeans< _TReal >::
init_Forgy(
  const Eigen::EigenBase< _TInput >& Ib,
  const unsigned long long& K
  )
{
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TInput >
void ivqML::Common::KMeans< _TReal >::
fit( const Eigen::EigenBase< _TInput >& Ib )
{
  auto I = Ib.derived( ).template cast< TReal >( );
  unsigned long long K = this->m_Means[ this->m_ActiveMean ].rows( );
  TMatrix D( I.rows( ), K ), J( I.rows( ), 1 );
  Eigen::Matrix< unsigned long long, Eigen::Dynamic, 1 > L( I.rows( ) );

  unsigned short e = 0;
  TReal mse = std::numeric_limits< TReal >::infinity( );
  bool stop = false;
  while( !stop )
  {
    // Update distances
    for( unsigned long long k = 0; k < K; ++k )
      D.col( k )
        =
        ( I.rowwise( ) - this->m_Means[ this->m_ActiveMean ].row( k ) ).
        array( ).pow( 2 ).rowwise( ).sum( ).sqrt( );

    // Update labels
    for( unsigned long long s = 0; s < L.rows( ); ++s )
      D.row( s ).minCoeff( &L( s, 0 ) );

    // Update means
    for( unsigned long long k = 0; k < K; ++k )
    {
      J = ( L.array( ) == k ).template cast< TReal >( );
      this->m_Means[ ( this->m_ActiveMean + 1 ) % 2 ].row( k )
        =
        ( I.array( ).colwise( ) * J.col( 0 ).array( ) ).colwise( ).sum( )
        /
        TReal( J.sum( ) );
    } // end for

    // Update error
    mse
      =
      (
        this->m_Means[ this->m_ActiveMean ]
        -
        this->m_Means[ ( this->m_ActiveMean + 1 ) % 2 ]
        ).array( ).pow( 2 ).rowwise( ).sum( ).mean( );
    this->m_ActiveMean = ( this->m_ActiveMean + 1 ) % 2;

    stop = ( this->m_Debug( mse ) || !( this->m_EPS < mse ) );
  } // end while
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TOutput, class _TInput >
void ivqML::Common::KMeans< _TReal >::
label(
  Eigen::EigenBase< _TOutput >& Lb,
  const Eigen::EigenBase< _TInput >& Ib
  )
{
  auto I = Ib.derived( ).template cast< TReal >( );
  auto L = Lb.derived( );
  unsigned long long K = this->m_Means[ this->m_ActiveMean ].rows( );
  TMatrix D( I.rows( ), K ), J( I.rows( ), 1 );

  // Compute distances
  for( unsigned long long k = 0; k < K; ++k )
    D.col( k )
      =
      ( I.rowwise( ) - this->m_Means[ this->m_ActiveMean ].row( k ) ).
      array( ).pow( 2 ).rowwise( ).sum( ).sqrt( );

  // Update labels
  for( unsigned long long s = 0; s < L.rows( ); ++s )
    D.row( s ).minCoeff( &L( s, 0 ) );
}

#endif // __ivqML__Common__KMeans__hxx__

// eof - $RCSfile$
