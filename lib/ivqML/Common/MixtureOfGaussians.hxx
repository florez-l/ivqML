// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Common__MixtureOfGaussians__hxx__
#define __ivqML__Common__MixtureOfGaussians__hxx__

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

// -------------------------------------------------------------------------
template< class _TReal >
ivqML::Common::MixtureOfGaussians< _TReal >::
MixtureOfGaussians( )
{
  this->m_Debug = []( const TReal& mse ) -> bool { return( false ); };
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Common::MixtureOfGaussians< _TReal >::
set_debug( TDebug d )
{
  this->m_Debug = d;
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TInput >
void ivqML::Common::MixtureOfGaussians< _TReal >::
init_random(
  const Eigen::EigenBase< _TInput >& Ib, const unsigned long long& K
  )
{
  this->_reserve( Ib.cols( ), K );

  auto I = Ib.derived( ).template cast< TReal >( );
  std::vector< unsigned long long > idx( I.rows( ) );
  std::iota( idx.begin( ), idx.end( ), 0 );
  std::shuffle(
    idx.begin( ), idx.end( ),
    std::default_random_engine(
      std::chrono::system_clock::now( ).time_since_epoch( ).count( )
      )
    );
  idx.erase( idx.begin( ) + K, idx.end( ) );

  this->m_Means = I( idx, ivq_EIGEN_ALL );
  this->_C( I );
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TInput >
void ivqML::Common::MixtureOfGaussians< _TReal >::
init_XX(
  const Eigen::EigenBase< _TInput >& Ib,
  const unsigned long long& K
  )
{
  this->_reserve( Ib.cols( ), K );
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TInput >
void ivqML::Common::MixtureOfGaussians< _TReal >::
init_Forgy(
  const Eigen::EigenBase< _TInput >& Ib,
  const unsigned long long& K
  )
{
  this->_reserve( Ib.cols( ), K );
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TInput >
void ivqML::Common::MixtureOfGaussians< _TReal >::
fit( const Eigen::EigenBase< _TInput >& Ib )
{
  auto I = Ib.derived( ).template cast< TReal >( );
  unsigned long long K = this->m_Means.rows( );
  unsigned long long F = I.cols( );

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
      this->m_COVs.block( k * F, 0, F, F )
        =
        ( c.transpose( ) * c ) / R.col( k ).sum( );
    } // end for

    // Stop criteria
    TReal e = this->_E( pp );
    std::cout << e << std::endl;
    std::cout << "------------------------" << std::endl;
    pp = this->m_Parameters;
  } // end while

  /* TODO
     using _TLabels = Eigen::Matrix< TInt, Eigen::Dynamic, 1 >;
     _TLabels L[ 2 ];
     L[ 0 ] = L[ 1 ] = _TLabels::Zero( I.rows( ) );
     unsigned long long iter = 0;
     TReal pl = std::numeric_limits< TReal >::max( );
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
template< class _TReal >
template< class _TOutput, class _TInput >
void ivqML::Common::MixtureOfGaussians< _TReal >::
label(
  Eigen::EigenBase< _TOutput >& Lb,
  const Eigen::EigenBase< _TInput >& Ib
  )
{
  const _TInput& I = Ib.derived( );
  _TOutput& L = Lb.derived( );

  TMatrix R( I.rows( ), this->m_Means.rows( ) );
  this->_R( R, I );
  this->_L( L, R );
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Common::MixtureOfGaussians< _TReal >::
_reserve( const TInt& F, const TInt& K )
{
  unsigned long long KF = K * F;
  this->m_Parameters = TBuffer::Zero( K + ( KF * ( 1 + F ) ) );

  TReal* d = this->m_Parameters.data( );
  new ( &( this->m_Weights ) ) MMatrix( d, 1, K );
  new ( &( this->m_Means ) )   MMatrix( d + K, K, F );
  new ( &( this->m_COVs ) )    MMatrix( d + ( K + KF ), KF, F );
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TInput >
void ivqML::Common::MixtureOfGaussians< _TReal >::
_C( const _TInput& I )
{
  unsigned long long K = this->m_Means.rows( );
  unsigned long long F = I.cols( );
  TMatrix D( I.rows( ), K ), J( I.rows( ), 1 );
  Eigen::Matrix< unsigned long long, Eigen::Dynamic, 1 > L( I.rows( ) );

  // Initial weights
  this->m_Weights = TBuffer::Ones( K ) / TReal( K );

  // Compute distances
  for( unsigned long long k = 0; k < K; ++k )
    D.col( k )
      =
      ( I.rowwise( ) - this->m_Means.row( k ) ).
      array( ).pow( 2 ).rowwise( ).sum( ).sqrt( );

  // Compute labels
  for( unsigned long long s = 0; s < L.rows( ); ++s )
    D.row( s ).minCoeff( &L( s ) );

  // Update means and initialize covariances
  this->m_COVs = TMatrix::Zero( F * K, F );
  for( unsigned long long k = 0; k < K; ++k )
  {
    J = ( L.array( ) == k ).template cast< TReal >( );
    auto P = ( I.array( ).colwise( ) * J.col( 0 ).array( ) ).eval( );

    unsigned long long a = J.sum( );
    this->m_Means.row( k ) = P.colwise( ).sum( ) / TReal( a );

    auto S = P.matrix( ).rowwise( ) - this->m_Means.row( k );
    this->m_COVs.block( k * F, 0, F, F )
      =
      ( S.transpose( ) * S ) / TReal( a - 1 );
  } // end for
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TInput >
_TReal ivqML::Common::MixtureOfGaussians< _TReal >::
_R( TMatrix& R, const _TInput& I ) const
{
  unsigned long long K = this->m_Means.rows( );
  unsigned long long F = I.cols( );
  TReal d = std::pow( TReal( 8 ) * std::atan( TReal( 1 ) ), F );

  for( unsigned long long k = 0; k < K; ++k )
  {
    auto C = this->m_COVs.block( k * F, 0, F, F );
    auto c = I.rowwise( ) - this->m_Means.row( k );
    R.col( k )
      =
      (
        ( c * C.inverse( ) * c.transpose( ) ).diagonal( ).array( )
        *
        TReal( -0.5 )
        ).array( ).exp( )
      *
      ( this->m_Weights( 0, k ) / std::sqrt( d * C.determinant( ) ) );
  } // end for

  TReal log_likelihood =
    ( R.rowwise( ).maxCoeff( ).array( ) + this->m_EPS ).log( ).sum( );
  R.array( ).colwise( ) /= R.array( ).rowwise( ).sum( );
  return( -log_likelihood );
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TOutput >
void ivqML::Common::MixtureOfGaussians< _TReal >::
_L( _TOutput& L, const TMatrix& R ) const
{
  Eigen::Index i;
  for( unsigned long long s = 0; s < L.rows( ); ++s )
  {
    R.row( s ).maxCoeff( &i );
    L( s ) = i;
  } // end for
}

#endif // __ivqML__Common__MixtureOfGaussians__hxx__

// eof - $RCSfile$
