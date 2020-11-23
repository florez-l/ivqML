// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <cassert>
#include <cmath>
#include <random>
#include "ClassificationTrainer.h"

// -------------------------------------------------------------------------
template< class _TANN >
ClassificationTrainer< _TANN >::
ClassificationTrainer( TNeuralNetwork* ann, const TScalar& epsilon )
  : m_Net( ann ),
    m_Alpha( TScalar( 1e-2 ) ),
    m_Lambda( TScalar( 0 ) ),
    m_Epsilon( epsilon ),
    m_BatchSize( 0 ),
    m_Train( 0.7 ),
    m_Test( 0.2 )
{
  this->setNormalizationToNone( );
}

// -------------------------------------------------------------------------
template< class _TANN >
ClassificationTrainer< _TANN >::
ClassificationTrainer( const Self& other )
{
  this->m_Net = other.m_Net;
  this->m_Alpha = other.m_Alpha;
  this->m_Lambda = other.m_Lambda;
  this->m_Epsilon = other.m_Epsilon;
  this->m_Train = other.m_Train;
  this->m_Test = other.m_Test;
  this->m_NormalizationType = other.m_NormalizationType;
}

// -------------------------------------------------------------------------
template< class _TANN >
typename ClassificationTrainer< _TANN >::
Self& ClassificationTrainer< _TANN >::
operator=( const Self& other )
{
  this->m_Net = other.m_Net;
  this->m_Alpha = other.m_Alpha;
  this->m_Lambda = other.m_Lambda;
  this->m_Epsilon = other.m_Epsilon;
  this->m_Train = other.m_Train;
  this->m_Test = other.m_Test;
  this->m_NormalizationType = other.m_NormalizationType;
  return( *this );
}

// -------------------------------------------------------------------------
template< class _TANN >
const typename ClassificationTrainer< _TANN >::
TScalar& ClassificationTrainer< _TANN >::
epsilon( ) const
{
  return( this->m_Epsilon );
}

// -------------------------------------------------------------------------
template< class _TANN >
const typename ClassificationTrainer< _TANN >::
TScalar& ClassificationTrainer< _TANN >::
learningRate( ) const
{
  return( this->m_Alpha );
}

// -------------------------------------------------------------------------
template< class _TANN >
const typename ClassificationTrainer< _TANN >::
TScalar& ClassificationTrainer< _TANN >::
regularization( ) const
{
  return( this->m_Lambda );
}

// -------------------------------------------------------------------------
template< class _TANN >
const unsigned long& ClassificationTrainer< _TANN >::
batchSize( ) const
{
  return( this->m_BatchSize );
}

// -------------------------------------------------------------------------
template< class _TANN >
bool ClassificationTrainer< _TANN >::
isDataNormalized( ) const
{
  return( this->m_NormalizationType == Self::None );
}

// -------------------------------------------------------------------------
template< class _TANN >
bool ClassificationTrainer< _TANN >::
isDataRescaled( ) const
{
  return( this->m_NormalizationType == Self::Rescale );
}

// -------------------------------------------------------------------------
template< class _TANN >
bool ClassificationTrainer< _TANN >::
isDataStandardized( ) const
{
  return( this->m_NormalizationType == Self::Standardization );
}

// -------------------------------------------------------------------------
template< class _TANN >
bool ClassificationTrainer< _TANN >::
isDataDecorrelated( ) const
{
  return( this->m_NormalizationType == Self::Decorrelation );
}

// -------------------------------------------------------------------------
template< class _TANN >
const typename ClassificationTrainer< _TANN >::
TScalar& ClassificationTrainer< _TANN >::
FtrainScore( ) const
{
  return( this->m_FtrainScore );
}

// -------------------------------------------------------------------------
template< class _TANN >
const typename ClassificationTrainer< _TANN >::
TScalar& ClassificationTrainer< _TANN >::
FtestScore( ) const
{
  return( this->m_FtestScore );
}

// -------------------------------------------------------------------------
template< class _TANN >
const typename ClassificationTrainer< _TANN >::
TScalar& ClassificationTrainer< _TANN >::
FvalidScore( ) const
{
  return( this->m_FvalidScore );
}

// -------------------------------------------------------------------------
template< class _TANN >
void ClassificationTrainer< _TANN >::
setEpsilon( const TScalar& e )
{
  this->m_Epsilon = e;
}

// -------------------------------------------------------------------------
template< class _TANN >
void ClassificationTrainer< _TANN >::
setLearningRate( const TScalar& a )
{
  this->m_Alpha = a;
}

// -------------------------------------------------------------------------
template< class _TANN >
void ClassificationTrainer< _TANN >::
setRegularization( const TScalar& l )
{
  this->m_Lambda = l;
}

// -------------------------------------------------------------------------
template< class _TANN >
void ClassificationTrainer< _TANN >::
setBatchSize( const unsigned long& s )
{
  this->m_BatchSize = s;
}

// -------------------------------------------------------------------------
template< class _TANN >
void ClassificationTrainer< _TANN >::
setSizes( const double& train, const double& test )
{
  assert(
    train >= 0.0 && test >= 0.0 && train + test <= 1.0 &&
    "Invalid sizes"
    );

  this->m_Train = train;
  this->m_Test = test;
}

// -------------------------------------------------------------------------
template< class _TANN >
void ClassificationTrainer< _TANN >::
setData( const TMatrix& X, const TMatrix& Y )
{
  assert( X.cols( ) == Y.cols( ) && "Incompatible sizes" );

  this->m_X = X;
  this->m_Y = Y;
}

// -------------------------------------------------------------------------
template< class _TANN >
void ClassificationTrainer< _TANN >::
setNormalizationToNone( )
{
  this->m_NormalizationType = Self::None;
}

// -------------------------------------------------------------------------
template< class _TANN >
void ClassificationTrainer< _TANN >::
setNormalizationToRescale( )
{
  this->m_NormalizationType = Self::Rescale;
}

// -------------------------------------------------------------------------
template< class _TANN >
void ClassificationTrainer< _TANN >::
setNormalizationToStandardization( )
{
  this->m_NormalizationType = Self::Standardization;
}

// -------------------------------------------------------------------------
template< class _TANN >
void ClassificationTrainer< _TANN >::
setNormalizationToDecorrelation( )
{
  this->m_NormalizationType = Self::Decorrelation;
}

// -------------------------------------------------------------------------
template< class _TANN >
void ClassificationTrainer< _TANN >::
train( const TTrainObserver& observer )
{
  using _TPerm = Eigen::PermutationMatrix< Eigen::Dynamic, Eigen::Dynamic >;

  if( this->m_Net == nullptr )
    return;

  // Sizes
  unsigned long m = this->m_X.cols( );
  unsigned long n = this->m_X.rows( );
  unsigned long p = this->m_Y.rows( );
  unsigned long m_train = ( unsigned long )( double( m ) * this->m_Train );
  unsigned long m_test = ( unsigned long )( double( m ) * this->m_Test );
  unsigned long m_valid = m - m_train - m_test;

  // Separate and normalize data
  this->m_Xtrain = this->m_X.block( 0, 0, n, m_train );
  this->m_Ytrain = this->m_Y.block( 0, 0, p, m_train );
  this->m_Xtest = this->m_X.block( 0, m_train, n, m_test );
  this->m_Ytest = this->m_Y.block( 0, m_train, p, m_test );
  this->m_Xvalid = this->m_X.block( 0, m_train + m_test, n, m_valid );
  this->m_Yvalid = this->m_Y.block( 0, m_train + m_test, p, m_valid );
  this->_normalize( );

  // Train
  this->_train( observer );

  // Confusion matrices
  this->m_FtrainScore = this->_f1_score( this->m_Xtrain, this->m_Ytrain );
  this->m_FtestScore  = this->_f1_score( this->m_Xtest, this->m_Ytest );
  this->m_FvalidScore = this->_f1_score( this->m_Xvalid, this->m_Yvalid );

  // Finish configuration
  this->m_Net->setNormalizationOffset( this->m_NormalizationOffset );
  this->m_Net->setNormalizationScale( this->m_NormalizationScale );
}

// -------------------------------------------------------------------------
template< class _TANN >
void ClassificationTrainer< _TANN >::
_normalize( )
{
  switch( this->m_NormalizationType )
  {
  case Self::Rescale:
    this->m_NormalizationOffset = this->m_Xtrain.rowwise( ).minCoeff( );
    break;
  case Self::Standardization:
    this->m_NormalizationOffset = -this->m_Xtrain.rowwise( ).mean( );
    this->m_NormalizationScale =
      TColVector(
        ( ( this->m_Xtrain.colwise( ) + this->m_NormalizationOffset ).
          array( ).square( ).rowwise( ).sum( ) /
          TScalar( this->m_Xtrain.cols( ) ) ).sqrt( )
        ).asDiagonal( ).inverse( );
    break;
  case Self::Decorrelation:
    // TODO: off = this->m_Xtrain.rowwise( ).mean( );
    this->m_NormalizationOffset = TMatrix::Zero( this->m_Xtrain.rows( ), 1 );
    this->m_NormalizationScale =
      TMatrix::Identity( this->m_Xtrain.rows( ), this->m_Xtrain.rows( ) );
    break;
  default:
    this->m_NormalizationOffset = TMatrix::Zero( this->m_Xtrain.rows( ), 1 );
    this->m_NormalizationScale =
      TMatrix::Identity( this->m_Xtrain.rows( ), this->m_Xtrain.rows( ) );
    break;
  } // end switch

  // Normalize data
  this->m_Xtrain =
    this->m_NormalizationScale *
    ( this->m_Xtrain.colwise( ) + this->m_NormalizationOffset );
  if( this->m_Xtest.cols( ) > 0 )
    this->m_Xtest =
      this->m_NormalizationScale *
      ( this->m_Xtest.colwise( ) + this->m_NormalizationOffset );
  if( this->m_Xvalid.cols( ) > 0 )
    this->m_Xvalid =
      this->m_NormalizationScale *
      ( this->m_Xvalid.colwise( ) + this->m_NormalizationOffset );
}

// -------------------------------------------------------------------------
template< class _TANN >
void ClassificationTrainer< _TANN >::
_train( const TTrainObserver& observer )
{
  // Prepare backprop objects and values
  long m = this->m_Xtrain.cols( );
  long n = this->m_Xtrain.rows( );
  long p = this->m_Ytrain.rows( );
  long L = this->m_Net->number_of_layers( );
  std::vector< TMatrix > a( L + 1 ), z( L ), d( L );
  a.shrink_to_fit( );
  z.shrink_to_fit( );
  d.shrink_to_fit( );

  // Prepare batch objects and values
  long bs = this->m_BatchSize;
  if( bs <= 0 || bs > m )
    bs = m;
  unsigned long nb =
    ( unsigned long )( std::ceil( double( m ) / double( bs ) ) );
  Eigen::PermutationMatrix< Eigen::Dynamic, Eigen::Dynamic > P( m );
  P.setIdentity( );

  // Main loop
  TScalar Jtrain = ( this->m_Ytrain - this->m_Net->f( this->m_Xtrain ) ).
    colwise( ).squaredNorm( ).mean( );
  TScalar Jtest = 0;
  if( this->m_Xtest.cols( ) > 0 )
    Jtest = ( this->m_Ytest - this->m_Net->f( this->m_Xtest ) ).
      colwise( ).squaredNorm( ).mean( );
  unsigned long i = 0;
  bool stop = false;
  while( !stop )
  {
    // Prepare batch
    std::random_shuffle(
      P.indices( ).data( ), P.indices( ).data( ) + P.indices( ).size( )
      );
    TMatrix Xp = this->m_Xtrain * P;
    TMatrix Yp = this->m_Ytrain * P;

    // Process examples
    for( unsigned long b = 0; b < nb; ++b )
    {
      // Get batch
      unsigned long s = b * bs;
      unsigned long e = s + bs;
      if( e >= m )
        e = m;
      TMatrix Xb = Xp.block( 0, s, n, e - s );
      TMatrix Yb = Yp.block( 0, s, p, e - s );
      TScalar lr = this->m_Alpha / TScalar( Xb.cols( ) );

      // Feed fwd
      a[ 0 ] = Xb;
      this->m_Net->f( a, z );

      // Back prop
      d[ L - 1 ] =
        ( Yb - a[ L ] ).array( ) *
        this->m_Net->sigma( L - 1 )->d( z[ L - 1 ] ).array( );
      for( long l = L - 2; l >= 0; --l )
        d[ l ] =
          (
            this->m_Net->weights( l + 1 ).transpose( ) *
            d[ l + 1 ]
            ).array( ) * this->m_Net->sigma( l )->d( z[ l ] ).array( );

      // Gradient descent
      for( long l = 0; l < L; ++l )
      {
        this->m_Net->weights( l ) += ( d[ l ] * a[ l ].transpose( ) ) * lr;
        this->m_Net->biases( l ) += d[ l ].rowwise( ).sum( ) * lr;
      } // end for
    } // end for

    // Cost
    Jtrain = ( this->m_Ytrain - this->m_Net->f( this->m_Xtrain ) ).
      colwise( ).squaredNorm( ).mean( );
    if( this->m_Xtest.cols( ) > 0 )
      Jtest = ( this->m_Ytest - this->m_Net->f( this->m_Xtest ) ).
        colwise( ).squaredNorm( ).mean( );
    stop = ( i == 50000 ); // ( ( J - Jn ) <= this->m_Epsilon );
    observer( i++, Jtrain, Jtest );
  } // end while
  observer( i, Jtrain, Jtest );
}

// -------------------------------------------------------------------------
template< class _TANN >
typename ClassificationTrainer< _TANN >::
TScalar ClassificationTrainer< _TANN >::
_f1_score( const TMatrix& X, const TMatrix& Y )
{
  static const TScalar _0 = TScalar( 0 );
  static const TScalar _1 = TScalar( 1 );
  static const TScalar _2 = TScalar( 2 );

  unsigned long m = X.cols( );
  if( m > 0 )
  {
    TMatrix Fp, Yp;
    if( Y.rows( ) == 1 )
    {
      Fp = TMatrix::Zero( 2, m );
      Yp = TMatrix::Zero( 2, m );
      Fp.block( 1, 0, 1, m ) = this->m_Net->t( X );
      Fp.block( 0, 0, 1, m ).array( ) = _1 - Fp.block( 1, 0, 1, m ).array( );
      Yp.block( 1, 0, 1, m ) = Y;
      Yp.block( 0, 0, 1, m ) = _1 - Yp.block( 1, 0, 1, m ).array( );
    }
    else
    {
      Fp = this->m_Net->t( X );
      Yp = Y;
    } // end if

    TMatrix F = Yp * Fp.transpose( );
    unsigned long n_classes = F.rows( );
    auto weights = F.rowwise( ).sum( ) / F.sum( );
    TScalar f1 = _0;
    for( unsigned long c = 0; c < n_classes; ++c )
    {
      TScalar tp = F( c, c );
      TScalar fp = F.row( c ).sum( ) - tp;
      TScalar fn = F.col( c ).sum( ) - tp;
      TScalar p = ( ( tp + fp ) > _0 )? tp / ( tp + fp ): _0;
      TScalar r = ( ( tp + fn ) > _0 )? tp / ( tp + fn ): _0;
      if( ( p + r ) > _0 )
        f1 += _2 * p * r * weights( c ) / ( p + r );
    } // end for
    return( f1 );
  }
  else
    return( 0 );
}

// -------------------------------------------------------------------------
#include "NeuralNetwork.h"

template class ClassificationTrainer< NeuralNetwork< float > >;
template class ClassificationTrainer< NeuralNetwork< double > >;
template class ClassificationTrainer< NeuralNetwork< long double > >;

// eof - $RCSfile$
