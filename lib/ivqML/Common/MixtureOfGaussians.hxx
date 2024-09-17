// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Common__MixtureOfGaussians__hxx__
#define __ivqML__Common__MixtureOfGaussians__hxx__

#include <cmath>
#include <ivqML/Common/KMeans.h>

// -------------------------------------------------------------------------
template< class _TM, class _TX >
void ivqML::Common::MixtureOfGaussians::
RandomInit( Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X )
{
  ivqML::Common::KMeans::RandomInit( _m, _X );
}

// -------------------------------------------------------------------------
template< class _TM, class _TX >
void ivqML::Common::MixtureOfGaussians::
ForgyInit( Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X )
{
  ivqML::Common::KMeans::ForgyInit( _m, _X );
}

// -------------------------------------------------------------------------
template< class _TM, class _TX >
void ivqML::Common::MixtureOfGaussians::
XXInit( Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X )
{
  ivqML::Common::KMeans::XXInit( _m, _X );
}

// -------------------------------------------------------------------------
template< class _TM, class _TX >
void ivqML::Common::MixtureOfGaussians::
Init(
  Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TX >& _X,
  const std::string& method
  )
{
  ivqML::Common::KMeans::Init( _m, _X, method );
}

// -------------------------------------------------------------------------
template< class _TM, class _TC, class _TX >
void ivqML::Common::MixtureOfGaussians::
Fit(
  Eigen::EigenBase< _TM >& _m, Eigen::EigenBase< _TC >& _C,
  const Eigen::EigenBase< _TX >& _X,
  std::function< bool( const typename _TM::Scalar&, const unsigned long long& ) > debug
  )
{
  using _R = typename _TM::Scalar;
  using _M = Eigen::Matrix< _R, Eigen::Dynamic, Eigen::Dynamic >;

  static const _R eps = std::numeric_limits< _R >::epsilon( );
  const auto& X = _X.derived( ).template cast< _R >( );
  auto& m = _m.derived( );
  auto& C = _C.derived( );
  unsigned long long F = m.cols( );
  unsigned long long K = m.rows( );
  unsigned long long N = X.rows( );

  // Init covariances
  C = _M::Zero( K * F, F );
  for( unsigned long long k = 0; k < K; ++k )
    C.block( k * F, 0, F, F ) = _M::Identity( F, F );

  // Some auxiliary variables
  _M R( N, K ), Pm = m, PC = C;
  _M W = _M::Ones( 1, K ) * ( _R( 1 ) / _R( K ) );

  // Go!
  unsigned long long iter = 0;
  bool stop = false;
  while( !stop )
  {
    // E-step: compute responsibilities
    ivqML::Common::MixtureOfGaussians::_Responsibilities( R, X, W, m, C );

    // M-step: update weights with a ***PARANOIAC PONDERATION!***
    W = R.colwise( ).mean( );
    W.array( ) /= W.sum( );

    // M-step: update means
    m
      =
      ( R.transpose( ) * X ).array( ).colwise( )
      /
      R.colwise( ).sum( ).transpose( ).array( );


    // M-step: update covariances
    for( unsigned long long k = 0; k < K; ++k )
    {
      auto c
        =
        (
          ( X.rowwise( ) - m.row( k ) ).array( ).colwise( )
          *
          R.col( k ).array( )
          ).matrix( );
      C.block( k * F, 0, F, F ) = c.transpose( ) * c;

      _R sR = R.col( k ).sum( );
      if( sR != _R( 0 ) )
        C.block( k * F, 0, F, F ).array( ) /= sR;
    } // end for

    // Stop criterion
    iter++;
    _R mse
      =
      ( Pm - m ).array( ).pow( 2 ).sum( )
      +
      ( PC - C ).array( ).pow( 2 ).sum( );
    mse /= _R( m.size( ) + C.size( ) );
    stop = debug( mse, iter ) || !( eps < mse );
    Pm = m;
    PC = C;
  } // end while
}

// -------------------------------------------------------------------------
template< class _TL, class _TX, class _TM, class _TC >
void ivqML::Common::MixtureOfGaussians::
Label(
  Eigen::EigenBase< _TL >& _L,
  const Eigen::EigenBase< _TX >& _X,
  const Eigen::EigenBase< _TM >& _m, const Eigen::EigenBase< _TC >& _C
  )
{
  using _R = typename _TM::Scalar;
  using _M = Eigen::Matrix< _R, Eigen::Dynamic, Eigen::Dynamic >;

  const auto& X = _X.derived( ).template cast< _R >( );
  auto& L = _L.derived( );
  const auto& m = _m.derived( );
  const auto& C = _C.derived( );
  unsigned long long K = m.rows( );
  unsigned long long F = m.cols( );
  unsigned long long N = X.rows( );
  _M D( N, K );
  _M E = _M::Identity( F, F ) * std::numeric_limits< _R >::epsilon( );

  // Compute distances
  for( unsigned long long k = 0; k < K; ++k )
  {
    auto d = X.rowwise( ) - m.row( k );
    auto S = ( C.block( k * F, 0, F, F ) + E ).inverse( );
    D.col( k )
      =
      ( ( d * S ).array( ) * d.array( ) ).rowwise( ).sum( ).sqrt( );
  } // end if

  // Compute labels
  for( unsigned long long n = 0; n < N; ++n )
    D.row( n ).minCoeff( &L( n, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TR, class _TX, class _TW, class _TM, class _TC >
void ivqML::Common::MixtureOfGaussians::
_Responsibilities(
  Eigen::EigenBase< _TR >& _r,
  const Eigen::EigenBase< _TX >& _X,
  const Eigen::EigenBase< _TW >& _W,
  const Eigen::EigenBase< _TM >& _m,
  const Eigen::EigenBase< _TC >& _C
  )
{
  using _R = typename _TM::Scalar;
  using _M = Eigen::Matrix< _R, Eigen::Dynamic, Eigen::Dynamic >;

  auto& r = _r.derived( );
  const auto& X = _X.derived( );
  const auto& W = _W.derived( );
  const auto& m = _m.derived( );
  const auto& C = _C.derived( );

  // Some auxiliary values
  static const _R _2pi = _R( 8 ) * std::atan( _R( 1 ) );
  unsigned long long K = m.rows( );
  unsigned long long F = m.cols( );
  _R d = std::pow( _2pi, _R( F ) * _R( -0.5 ) );
  _M E = _M::Identity( F, F ) * std::numeric_limits< _R >::epsilon( );

  for( unsigned long long k = 0; k < K; ++k )
  {
    auto S = C.block( k * F, 0, F, F ) + E;
    _R D = S.determinant( );
    if( D > _R( 0 ) )
    {
      auto c = X.rowwise( ) - m.row( k );
      r.col( k )
        =
        (
          ( ( c * S.inverse( ) ).array( ) * c.array( ) ).rowwise( ).sum( )
          *
          _R( -0.5 )
          ).exp( )
        *
        ( W( 0, k ) * d / std::sqrt( D ) );
    }
    else
      r.col( k ).array( ) *= _R( 0 );
  } // end for

  auto s = r.array( ).rowwise( ).sum( );
  r.array( ).colwise( ) /= ( s == 0 ).select( 1, s );
}














































// -------------------------------------------------------------------------
/* TODO
template< class __R >
template< class _TInput >
__R ivqML::Common::MixtureOfGaussians< __R >::
_R( TMatrix& R, const _TInput& I ) const
{
}

// -------------------------------------------------------------------------
template< class __R >
template< class _TOutput >
void ivqML::Common::MixtureOfGaussians< __R >::
_L( _TOutput& L, const TMatrix& R ) const
{
  Eigen::Index i;
  for( unsigned long long s = 0; s < L.rows( ); ++s )
  {
    R.row( s ).maxCoeff( &i );
    L( s ) = i;
  } // end for
}
*/

#endif // __ivqML__Common__MixtureOfGaussians__hxx__

// eof - $RCSfile$
