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

    std::cout << W << std::endl;
    std::cout << W.sum( ) << std::endl;
    std::exit( 1 );

  } // end while

  /* TODO
     auto I = Ib.derived( ).template cast< _R >( );
     unsigned long long K = this->m_Means.rows( );
     unsigned long long F = this->m_Means.cols( );

     // Prepare iteration-related values
     TMatrix R( I.rows( ), K );
     TBuffer pp = TBuffer::Zero( this->m_Parameters.size( ) );
     bool stop = false;

     // Go!
     while( !stop )
     {
     // E-step: compute responsibilities
     this->_R( R, I );

    // M-step: update weights with a ***PARANOIAC PONDERATION!***
    this->m_Weights = R.colwise( ).mean( );
    this->m_Weights.array( ) /= this->m_Weights.sum( );

    // M-step: update means
    this->m_Means
      =
      ( R.transpose( ) * I ).array( ).colwise( )
      /
      R.colwise( ).sum( ).transpose( ).array( );

    // M-step: update covariances
    for( unsigned long long k = 0; k < K; ++k )
    {
      auto c
        =
        (
          ( I.rowwise( ) - this->m_Means.row( k ) ).array( ).colwise( )
          *
          R.col( k ).array( )
          ).matrix( );
      this->m_COVs.block( k * F, 0, F, F ) = c.transpose( ) * c;

      _R sR = R.col( k ).sum( );
      if( sR != _R( 0 ) )
        this->m_COVs.block( k * F, 0, F, F ).array( ) /= sR;
    } // end for

    // Stop criteria
    _R e = this->_E( pp );
    std::cout << e << std::endl;
    pp = this->m_Parameters;
  } // end while
  */

  /* TODO
     using _TLabels = Eigen::Matrix< TInt, Eigen::Dynamic, 1 >;
     _TLabels L[ 2 ];
     L[ 0 ] = L[ 1 ] = _TLabels::Zero( I.rows( ) );
     unsigned long long iter = 0;
     _R pl = std::numeric_limits< _R >::max( );
     bool stop = false;

     // Go!
     while( !stop )
     {
       this->_L( L[ ( iter + 1 ) % 2 ], R );
     unsigned long long count
     =
     ( L[ 0 ].array( ) != L[ 1 ].array( ) ).
     template cast< unsigned long long >( ).sum( );
     stop = ( count == 0 );
     this->m_Debug( pl - ll );
     pl = ll;
     iter++;
     } // end while
  */
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
  /* TODO
     const _TInput& I = Ib.derived( );
     _TOutput& L = Lb.derived( );

     TMatrix R( I.rows( ), this->m_Means.rows( ) );
     this->_R( R, I );
     this->_L( L, R );
  */
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
