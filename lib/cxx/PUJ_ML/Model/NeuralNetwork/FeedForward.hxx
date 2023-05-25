// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__NeuralNetwork__FeedForward__hxx__
#define __PUJ_ML__Model__NeuralNetwork__FeedForward__hxx__

#include <algorithm>
#include <cctype>
#include <random>
#include <sstream>

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
  unsigned long long L = this->m_W.size( );

  std::vector< unsigned long long > sizes( L + 1 );
  unsigned long long s = 0;
  sizes[ 0 ] = this->m_W[ 0 ].rows( );
  for( unsigned long long l = 0; l < L; ++l )
  {
    sizes[ l + 1 ] = this->m_W[ l ].cols( );
    s += this->m_W[ l ].cols( ) * ( this->m_W[ l ].rows( ) + 1 );
  } // end for
  this->m_P.clear( );
  this->m_W.clear( );
  this->m_B.clear( );
  std::random_device dev;
  std::default_random_engine eng( dev( ) );
  std::uniform_real_distribution< TReal > w( 1e-2, 1 );
  std::uniform_int_distribution< char > sgn( 0, 1 );
  for( unsigned long long i = 0; i < s; ++i )
    this->m_P.push_back( TReal( ( sgn( eng ) << 1 ) - 1 ) * w( eng ) );

  // Reassign memory
  TReal* data = this->m_P.data( );
  s = 0;
  for( unsigned long long l = 0; l < L; ++l )
  {
    unsigned long long i = sizes[ l ];
    unsigned long long o = sizes[ l + 1 ];

    this->m_W.push_back( MMatrix( data + s, i, o ) );
    this->m_B.push_back( MRow( data + s + ( i * o ), 1, o ) );

    s += ( i * o ) + o;
  } // end for
}

// -------------------------------------------------------------------------
template< class _R >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
add_layer(
  const unsigned long long& input,
  const unsigned long long& output,
  const std::string& activation
  )
{
  this->m_P.clear( );
  this->m_W.clear( );
  this->m_B.clear( );
  this->m_A.clear( );
  this->m_S.clear( );

  unsigned long long p = output * input;

  this->m_W.push_back( MMatrix( nullptr, input, output ) );
  this->m_B.push_back( MRow( nullptr, 1, output ) );
  this->m_A.push_back( Self::s_Activations( activation ) );
  this->m_S.push_back( activation );
}

// -------------------------------------------------------------------------
template< class _R >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
add_layer( const unsigned long long& output, const std::string& activation )
{
  if( this->m_W.size( ) > 0 )
  {
    unsigned long long input = this->m_W.back( ).cols( );
    unsigned long long n = this->m_P.size( );
    unsigned long long p = output * input;

    this->m_W.push_back( MMatrix( nullptr, input, output ) );
    this->m_B.push_back( MRow( nullptr, 1, output ) );
    this->m_A.push_back( Self::s_Activations( activation ) );
    this->m_S.push_back( activation );
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
threshold( Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X ) const
{
  std::string a = this->m_S.back( );
  std::transform(
    a.begin( ), a.end( ), a.begin( ),
    []( unsigned char c ){ return( std::tolower( c ) ); }
    );

  if( a == "softmax" )
  {
    TMatrix Z;
    this->evaluate( Z, X );
    Y.derived( ).resize( Z.rows( ), 1 );

    Eigen::Index i;
    for( unsigned long long r = 0; r < Z.rows( ); ++r )
    {
      Z.row( r ).maxCoeff( &i );
      Y.derived( )( r, 0 ) = ( typename _Y::Scalar )( i );
    } // end for
  }
  else if( a == "sigmoid" )
  {
    TMatrix Z;
    this->evaluate( Z, X );
    Y.derived( ) =
      Z.unaryExpr(
        [&]( TReal z ) -> typename _Y::Scalar
        {
          return( ( typename _Y::Scalar )( ( z < TReal( 0.5 ) )? 0: 1 ) );
        }
        );
  }
  else
    this->evaluate( Y, X );
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

  for( unsigned long long l = 0; l < this->m_W.size( ); ++l )
  {
    Z = ( A * this->m_W[ l ] ).rowwise( ) + this->m_B[ l ];
    this->m_A[ l ]( A, Z, false );

    if( As != nullptr ) As->push_back( A );
    if( Zs != nullptr ) Zs->push_back( Z );
  } // end for
  Y.derived( ) = A.template cast< typename _Y::Scalar >( );
}

// -------------------------------------------------------------------------
template< class _R >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
_from_stream( std::istream& i )
{
  // Network's topology
  unsigned long long L, in, out;
  std::string activation;
  i >> L >> in;
  for( unsigned long long l = 0; l < L; ++l )
  {
    i >> out >> activation;
    if( l == 0 )
      this->add_layer( in, out, activation );
    else
      this->add_layer( out, activation );
  } // end for
  this->init( );

  // Weights
  std::string n_params;
  i >> n_params;
  std::transform(
    n_params.begin( ), n_params.end( ), n_params.begin( ),
    []( unsigned char c ){ return( std::tolower( c ) ); }
    );
  if( n_params != "random" )
  {
    std::istringstream n_params_str( n_params );
    unsigned long long N;
    n_params_str >> N;
    if( this->m_P.size( ) == N )
      for( unsigned long long n = 0; n < N; ++n )
        i >> this->m_P[ n ];
  } // end if
}

// -------------------------------------------------------------------------
template< class _R >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
_to_stream( std::ostream& o ) const
{
  o << this->m_W.size( ) << " " << this->m_W[ 0 ].rows( ) << std::endl;
  for( unsigned long long l = 0; l < this->m_W.size( ); ++l )
    o
      << this->m_W[ l ].cols( ) << " "
      << this->m_S[ l ] << std::endl;
  this->Superclass::_to_stream( o );
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
