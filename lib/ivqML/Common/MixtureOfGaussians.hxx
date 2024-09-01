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
  this->m_Means = I( means, ivq_EIGEN_ALL );
  /* TODO
     this->m_Means[ 1 ]
     =
     TMatrix::Zero( this->m_Means[ 0 ].rows( ), this->m_Means[ 0 ].cols( ) );
     this->m_ActiveMean = 0;
  */
  this->_init_COVs( I );
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
  TMatrix R( I.rows( ), K );

  bool stop = false;
  TReal prev_likelihood = std::numeric_limits< TReal >::lowest( );
  while( !stop )
  {
    // E-step: compute responsibilities
    this->_PDF( Ib, R );
    TReal likelihood = ( R.colwise( ).maxCoeff( ).array( ) + this->m_EPS ).log( ).sum( );
    // stop = prev_likelihood - likelihood;

    std::cout << likelihood << std::endl;
    prev_likelihood = likelihood;

    R.array( ).colwise( ) /= R.array( ).rowwise( ).sum( );

    // M-step: update parameters
    this->m_Weights = R.colwise( ).mean( );
    this->m_Weights.array( ) /= this->m_Weights.sum( ); // PARANOIAC PONDERATION!
    this->m_Means
      =
      ( R.transpose( ) * I ).array( ).colwise( )
      /
      R.colwise( ).sum( ).transpose( ).array( );

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
  } // end while

  /* TODO
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
  // Responsibilities
  TMatrix R( Ib.rows( ), this->m_Means.rows( ) );
  this->_PDF( Ib, R );
  R.array( ).colwise( ) /= R.array( ).rowwise( ).sum( );

  // Compute labels
  auto L = Lb.derived( );
  for( unsigned long long s = 0; s < L.rows( ); ++s )
    R.row( s ).maxCoeff( &L( s, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TInput >
void ivqML::Common::MixtureOfGaussians< _TReal >::
_init_COVs( const Eigen::EigenBase< _TInput >& Ib )
{
  auto I = Ib.derived( ).template cast< TReal >( );
  unsigned long long K = this->m_Means.rows( );
  unsigned long long F = I.cols( );
  TMatrix D( I.rows( ), K ), J( I.rows( ), 1 );
  Eigen::Matrix< unsigned long long, Eigen::Dynamic, 1 > L( I.rows( ) );

  // Compute distances
  for( unsigned long long k = 0; k < K; ++k )
    D.col( k )
      =
      ( I.rowwise( ) - this->m_Means.row( k ) ).
      array( ).pow( 2 ).rowwise( ).sum( ).sqrt( );

  // Compute labels
  for( unsigned long long s = 0; s < L.rows( ); ++s )
    D.row( s ).minCoeff( &L( s, 0 ) );

  // Update means and COVs
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

  // Update weights
  this->m_Weights = TMatrix::Ones( 1, K ) / TReal( K );
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TInput >
void ivqML::Common::MixtureOfGaussians< _TReal >::
_PDF( const Eigen::EigenBase< _TInput >& Ib, TMatrix& R )
{
  auto I = Ib.derived( ).template cast< TReal >( );
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
}

#endif // __ivqML__Common__MixtureOfGaussians__hxx__

// eof - $RCSfile$




// ... (rest of the GaussianMixtureModel class)

/* TODO
double log_likelihood(const Eigen::MatrixXd& X) const {
    int N = X.rows();
    double log_likelihood = 0.0;

    for (int i = 0; i < N; ++i) {
        double max_component_prob = -std::numeric_limits<double>::max();
        for (int k = 0; k < n_components; ++k) {
            double component_prob = weights(k) * multivariate_normal_pdf(X.row(i), means.row(k), covariances.block<n_features, n_features>(k, 0));
            max_component_prob = std::max(max_component_prob, component_prob);
        }

        log_likelihood += std::log(max_component_prob);
    }

    return log_likelihood;
}

*/
