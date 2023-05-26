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
number_of_layers( ) const
{
  return( this->m_A.size( ) );
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
  if( l < this->m_S.size( ) )
    return( this->m_S[ l ] );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _R >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
init( const unsigned long long& n )
{
  unsigned long long s = 0;
  for( unsigned long long l = 0; l < this->m_A.size( ); ++l )
    s += this->m_S[ l ] * ( this->m_S[ l + 1 ] + 1 );
  this->Superclass::init( s );

  std::random_device dev;
  std::default_random_engine eng( dev( ) );
  std::uniform_real_distribution< TReal > w( 1e-2, 1 );
  std::uniform_int_distribution< char > sgn( 0, 1 );
  for( unsigned long long i = 0; i < s; ++i )
    this->m_P[ i ] = TReal( ( sgn( eng ) << 1 ) - 1 ) * w( eng );
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
  this->m_S.clear( );
  this->m_A.clear( );

  std::string a = activation;
  std::transform(
    a.begin( ), a.end( ), a.begin( ),
    []( unsigned char c ){ return( std::tolower( c ) ); }
    );

  this->m_S.push_back( input );
  this->m_S.push_back( output );
  this->m_A.push_back( std::make_pair( a, Self::s_Activations( a ) ) );
}

// -------------------------------------------------------------------------
template< class _R >
void PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
add_layer( const unsigned long long& output, const std::string& activation )
{
  if( this->m_S.size( ) > 1 )
  {
    std::string a = activation;
    std::transform(
      a.begin( ), a.end( ), a.begin( ),
      []( unsigned char c ){ return( std::tolower( c ) ); }
      );

    this->m_S.push_back( output );
    this->m_A.push_back( std::make_pair( a, Self::s_Activations( a ) ) );
  }
  else
    this->add_layer( output, output, activation );
}

// -------------------------------------------------------------------------
template< class _R >
typename PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
ConstMMatrix PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
weights( unsigned long long L ) const
{
  unsigned long long s = 0, i, o;
  for( unsigned long long l = 0; l < L; ++l )
  {
    i = this->m_S[ l ];
    o = this->m_S[ l + 1 ];
    s += o * ( i + 1 );
  } // end for
  s -= o * ( i + 1 );
  return( ConstMMatrix( this->m_P.data( ) + s, i, o ) );
}

// -------------------------------------------------------------------------
template< class _R >
const std::pair< std::string, typename PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::TActivation >&
PUJ_ML::Model::NeuralNetwork::FeedForward< _R >::
activation( unsigned long long l ) const
{
  return( this->m_A[ l ] );
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
  std::string a = this->m_A.back( ).first;
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

  unsigned long long s = 0;
  const TReal* d = this->m_P.data( );
  for( unsigned long long l = 0; l < this->m_A.size( ); ++l )
  {
    unsigned long long i = this->m_S[ l ];
    unsigned long long o = this->m_S[ l + 1 ];

    Z =
      ( A * ConstMMatrix( d + s, i, o ) ).rowwise( )
      +
      ConstMRow( d + s + ( i * o ), 1, o );
    this->m_A[ l ].second( A, Z, false );

    if( As != nullptr ) As->push_back( A );
    if( Zs != nullptr ) Zs->push_back( Z );

    s += o * ( i + 1 );
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
  o << this->m_A.size( ) << " " << this->m_S[ 0 ] << std::endl;
  for( unsigned long long l = 0; l < this->m_A.size( ); ++l )
    o
      << this->m_S[ l + 1 ] << " "
      << this->m_A[ l ].first << std::endl;
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
  const Eigen::EigenBase< _X >& X, const Eigen::EigenBase< _Y >& Y,
  TReal* G
  ) const
{
  auto iX = X.derived( ).template cast< TReal >( );
  auto iY = Y.derived( ).template cast< TReal >( );

  // Forward propagation
  std::vector< TMatrix > As, Zs, Ds;
  TMatrix rY;
  this->m_Model->_evaluate( rY, iX, &As, &Zs );

  // Cost
  TReal J = 0;
  /* TODO
     # 2. Cost
      J = float( 0 )
      last_activation = str( self.m_Model.m_S[ -1 ] )
      if last_activation == 'SoftMax':
        for k in range( Y.shape[ 1 ] ):
          J -= numpy.log( A[ -1 ][ : , Y[ : , k ] == 1 ] + 1e-12 ).sum( )
          J -= numpy.log( 1.0 - A[ -1 ][ : , Y[ : , k ] == 0 ] + 1e-12 ).sum( )
        # end for
        J /= float( X.shape[ 0 ] )
      elif last_activation == 'Sigmoid':
        a = A[ -1 ].T
        J -= numpy.log( a[ Y == 1 ] + 1e-12 ).sum( )
        J -= numpy.log( 1 - a[ Y == 0 ] + 1e-12 ).sum( )
        J /= float( X.shape[ 0 ] )
      else:
        pass
      # end if
  */

  // Backpropagation
  if( G != nullptr )
  {
    unsigned long long L = this->m_Model->number_of_layers( );
    TReal m = TReal( iX.rows( ) );

    // Compute deltas
    std::string la = this->m_Model->activation( L - 1 ).first;
    if( la == "softmax" || la == "sigmoid" )
      Ds.push_back( ( As.back( ) - iY ) / m );
    else
    {
      // TODO
    } // end if
    for( unsigned long long l = 1; l < L; ++l )
    {
      auto B = Ds.back( ) * this->m_Model->weights( L - l + 1 ).transpose( );
      TMatrix dZ;
      this->m_Model->activation( L - l - 1 ).
        second( dZ, Zs[ L - l - 1 ], true );
      dZ.array( ) *= B.array( );
      Ds.push_back( dZ );
    } // end for

    // Compute derivatives
    unsigned long long s = 0;
    for( unsigned long long l = 0; l < L; ++l )
    {
      unsigned long long i = this->m_Model->number_of_inputs( l );
      unsigned long long o = this->m_Model->number_of_inputs( l + 1 );

      MMatrix( G + s, i, o ) = As[ l ].transpose( ) * Ds[ L - 1 - l ];
      MRow( G + s + ( i * o ), 1, o ) = Ds[ L - 1 - l ].colwise( ).mean( );

      s += o * ( i + 1 );
    } // end for
  } // end if
  return( J );
}

#endif // __PUJ_ML__Model__NeuralNetwork__FeedForward__hxx__

// eof - $RCSfile$
