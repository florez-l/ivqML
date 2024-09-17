// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Common__KMeans__hxx__
#define __ivqML__Common__KMeans__hxx__

#include <algorithm>
#include <cctype>
#include <numeric>
#include <random>
#include <vector>

// -------------------------------------------------------------------------
template< class _TM, class _TX >
void ivqML::Common::KMeans::
RandomInit( Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X )
{
  using _R = typename _TM::Scalar;

  std::vector< unsigned long long > I( _X.rows( ) );
  std::iota( I.begin( ), I.end( ), 0 );
  std::shuffle(
    I.begin( ), I.end( ),
    std::default_random_engine(
      std::chrono::system_clock::now( ).time_since_epoch( ).count( )
      )
    );
  I.erase( I.begin( ) + _m.rows( ), I.end( ) );
  _m.derived( ) = _X.derived( )( I, ivq_EIGEN_ALL ).template cast< _R >( );
}

// -------------------------------------------------------------------------
template< class _TM, class _TX >
void ivqML::Common::KMeans::
ForgyInit( Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X )
{
  ivqML::Common::KMeans::XXInit( _m, _X );
}

// -------------------------------------------------------------------------
template< class _TM, class _TX >
void ivqML::Common::KMeans::
XXInit( Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X )
{
  using _R = typename _TM::Scalar;
  using _M = Eigen::Matrix< _R, Eigen::Dynamic, Eigen::Dynamic >;

  const auto& X = _X.derived( ).template cast< _R >( );
  auto& m = _m.derived( );

  auto eng
    =
    std::default_random_engine(
      std::chrono::system_clock::now( ).time_since_epoch( ).count( )
      );
  std::uniform_int_distribution< unsigned long long > dis( 0, X.rows( ) - 1 );
  m.row( 0 ) = X.row( dis( eng ) );

  unsigned long long K = m.rows( );
  for( unsigned long long k = 1; k < K; ++k )
  {
    // Distances
    _M D( _X.rows( ), k );
    for( unsigned long long j = 0; j < k; ++j )
      D.col( j )
        =
        ( X.rowwise( ) - m.row( j ) )
        .array( ).pow( 2 ).rowwise( ).sum( ).sqrt( );

    // Next candidate
    unsigned long long idx;
    D.rowwise( ).minCoeff( ).maxCoeff( &idx );
    m.row( k ) = X.row( idx );
  } // end for
}

// -------------------------------------------------------------------------
template< class _TM, class _TX >
void ivqML::Common::KMeans::
Init(
  Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X,
  const std::string& method
  )
{
  std::string n;
  std::transform(
    method.begin( ), method.end( ), std::back_inserter( n ),
    []( unsigned char c ) -> unsigned char { return( std::tolower( c ) ); }
    );

  if     ( n == "random" ) ivqML::Common::KMeans::RandomInit( _m, _X );
  else if( n == "forgy" )  ivqML::Common::KMeans::ForgyInit( _m, _X );
  else                     ivqML::Common::KMeans::XXInit( _m, _X );
}

// -------------------------------------------------------------------------
template< class _TM, class _TX >
void ivqML::Common::KMeans::
Fit(
  Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X,
  std::function< bool( const typename _TM::Scalar&, const unsigned long long& ) > debug
  )
{
  using _R = typename _TM::Scalar;
  using _M = Eigen::Matrix< _R, Eigen::Dynamic, Eigen::Dynamic >;
  using _L = Eigen::Matrix< unsigned short, Eigen::Dynamic, 1 >;

  // Some auxiliary variables
  static const _R eps = std::numeric_limits< _R >::epsilon( );
  const auto& X = _X.derived( ).template cast< _R >( );
  auto& m = _m.derived( );
  unsigned long long F = m.cols( );
  unsigned long long K = m.rows( );
  unsigned long long N = X.rows( );
  _M D( N, K ), J( N, 1 ), P = m;
  _L L( N );

  // Go!
  unsigned long long iter = 0;
  bool stop = false;
  while( !stop )
  {
    // Update distances
    for( unsigned long long k = 0; k < K; ++k )
      D.col( k )
        =
        ( X.rowwise( ) - m.row( k ) )
        .array( ).pow( 2 ).rowwise( ).sum( ).sqrt( );

    // Update labels
    for( unsigned long long n = 0; n < N; ++n )
      D.row( n ).minCoeff( &L( n ) );

    // Update means
    for( unsigned long long k = 0; k < K; ++k )
    {
      J = ( L.array( ) == k ).template cast< _R >( );
      unsigned long long j = J.sum( );
      if( j > 0 )
        m.row( k )
          =
          ( X.array( ).colwise( ) * J.col( 0 ).array( ) ).colwise( ).sum( )
          /
          _R( j );
      else
        m.row( k ).array( ) *= _R( 0 );
    } // end for

    // Stop criterion
    iter++;
    _R mse = ( P - m ).array( ).pow( 2 ).mean( );
    stop = debug( mse, iter ) || !( eps < mse );
    P = m;
  } // end while
}

// -------------------------------------------------------------------------
template< class _TL, class _TX, class _TM >
void ivqML::Common::KMeans::
Label(
  Eigen::EigenBase< _TL >& _L,
  const Eigen::EigenBase< _TX >& _X,
  const Eigen::EigenBase< _TM >& _m
  )
{
  using _R = typename _TM::Scalar;
  using _M = Eigen::Matrix< _R, Eigen::Dynamic, Eigen::Dynamic >;

  const auto& X = _X.derived( ).template cast< _R >( );
  auto& L = _L.derived( );
  auto& m = _m.derived( );
  unsigned long long K = m.rows( );
  unsigned long long N = X.rows( );
  _M D( N, K );

  // Compute distances
  for( unsigned long long k = 0; k < K; ++k )
    D.col( k )
      =
      ( X.rowwise( ) - m.row( k ) )
      .array( ).pow( 2 ).rowwise( ).sum( ).sqrt( );

  // Compute labels
  for( unsigned long long n = 0; n < N; ++n )
    D.row( n ).minCoeff( &L( n, 0 ) );
}

#endif // __ivqML__Common__KMeans__hxx__

// eof - $RCSfile$
