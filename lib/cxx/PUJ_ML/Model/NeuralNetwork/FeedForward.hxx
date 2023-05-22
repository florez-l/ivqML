// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__NeuralNetwork__FeedForward__hxx__
#define __PUJ_ML__Model__NeuralNetwork__FeedForward__hxx__

#include <algorithm>
#include <random>

// -------------------------------------------------------------------------
template< class _R >
PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
FeedForward( const unsigned long long& n )
  : Superclass( n )
{
}

// -------------------------------------------------------------------------
template< class _R >
PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
~FeedForward( )
{
}

// -------------------------------------------------------------------------
template< class _R >
unsigned long long PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
number_of_inputs( ) const
{
  return( this->number_of_inputs( 0 ) );
}

// -------------------------------------------------------------------------
template< class _R >
unsigned long long PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
number_of_inputs( const unsigned long long& l ) const
{
  if( l < this->m_W.size( ) )
    return( this->m_W[ l ].rows( ) );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _R >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
init( const unsigned long long& n )
{
  std::random_device dev;
  std::default_random_engine eng( dev( ) );
  std::uniform_real_distribution< TReal > w( 1e-2, 1 );
  std::uniform_int_distribution< char > s( 0, 1 );
  std::transform(
    this->m_P.begin( ), this->m_P.end( ), this->m_P.begin( ),
    [&]( TReal v ) -> TReal
    {
      return( TReal( ( s( eng ) << 1 ) - 1 ) * w( eng ) );
    }
    );
}

// -------------------------------------------------------------------------
template< class _R >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
add_layer(
  const unsigned long long& input,
  const unsigned long long& output,
  TActivation activation
  )
{
  this->m_P.clear( );
  this->m_W.clear( );
  this->m_B.clear( );
  this->m_A.clear( );

  unsigned long long p = output * input;
  this->m_P.resize( p + output, 0 );
  this->m_P.shrink_to_fit( );

  this->m_W.push_back( MMatrix( this->m_P.data( ), input, output ) );
  this->m_B.push_back( MCol( this->m_P.data( ) + p, output, 1 ) );
  this->m_A.push_back( activation );
}

// -------------------------------------------------------------------------
template< class _R >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
add_layer( const unsigned long long& output, TActivation activation )
{
  if( this->m_W.size( ) > 0 )
  {
    unsigned long long input = this->m_W.back( ).cols( );
    unsigned long long n = this->m_P.size( );
    unsigned long long p = output * input;
    this->m_P.resize( n + p + output, 0 );
    this->m_P.shrink_to_fit( );

    this->m_W.push_back( MMatrix( this->m_P.data( ) + n, input, output ) );
    this->m_B.push_back( MCol( this->m_P.data( ) + n + p, output, 1 ) );
    this->m_A.push_back( activation );
  }
  else
    this->add_layer( output, output, activation );
}

// -------------------------------------------------------------------------
template< class _R >
template< class _Y, class _X >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
evaluate( Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X ) const
{
  this->_evaluate( Y, X );
}

// -------------------------------------------------------------------------
template< class _R >
template< class _Y, class _X >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
_evaluate(
  Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X,
  std::vector< TMatrix >* As, std::vector< TMatrix >* Zs
  ) const
{
  TMatrix A = X.derived( ).template cast< TReal >( ), Z;

  if( As != nullptr )
  {
    As->clear( );
    As->push_back( A );
  } // end if
  if( Zs != nullptr )
    Zs->clear( );

  for( unsigned long long i = 0; i < this->m_W.size( ); ++i )
  {
    Z = ( A * this->m_W[ i ] ) + this->m_B[ i ];
    this->m_A( A, Z, false );

    if( As != nullptr ) As->push_back( A );
    if( Zs != nullptr ) Zs->push_back( Z );
  } // end for
  Y.derived( ) = Z.template cast< typename _R::Scalar >( );
}

// -------------------------------------------------------------------------
template< class _R >
PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::Cost::
Cost( TModel* m )
  : m_Model( m )
{
}

// -------------------------------------------------------------------------
template< class _R >
template< class _X, class _Y >
typename PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::TReal
PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::Cost::
evaluate(
  const Eigen::EigenBase< _X >& X, const Eigen::EigenBase< _Y >& Y
  ) const
{
  return( TReal( 0 ) );
}

// -------------------------------------------------------------------------
template< class _R >
template< class _X, class _Y >
typename PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::TReal
PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::Cost::
gradient(
  std::vector< TReal >& G,
  const Eigen::EigenBase< _X >& X,
  const Eigen::EigenBase< _Y >& Y
  ) const
{
  return( TReal( 0 ) );
}

#endif // __PUJ_ML__Model__NeuralNetwork__FeedForward__hxx__

// eof - $RCSfile$
