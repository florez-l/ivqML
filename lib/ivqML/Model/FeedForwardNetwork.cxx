// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include <ivqML/Model/FeedForwardNetwork.h>

// -------------------------------------------------------------------------
template< class _S >
ivqML::Model::FeedForwardNetwork< _S >::
FeedForwardNetwork( )
  : Superclass( TNatural( 0 ) )
{
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
random_fill( )
{
  this->Superclass::random_fill( );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::FeedForwardNetwork< _S >::
TNatural ivqML::Model::FeedForwardNetwork< _S >::
number_of_inputs( ) const
{
  if( this->m_S.size( ) > 0 )
    return( this->m_S[ 0 ] );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
set_number_of_inputs( const TNatural& p )
{
  // WARNING: do nothing!
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::FeedForwardNetwork< _S >::
TNatural ivqML::Model::FeedForwardNetwork< _S >::
number_of_outputs( ) const
{
  if( this->m_S.size( ) > 0 )
    return( this->m_S.back( ) );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
set_number_of_parameters( const TNatural& p )
{
  // WARNING: do nothing!
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
add_layer( const TNatural& i, const TNatural& o, const std::string& a )
{
  this->m_S.clear( );
  this->m_F.clear( );
  this->m_W.clear( );
  this->m_B.clear( );
  this->m_F.clear( );

  this->m_S.push_back( i );
  this->m_S.push_back( o );

  this->m_F.push_back( std::make_pair( a, TActivationFactory::New( a ) ) );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
add_layer( const TNatural& o, const std::string& a )
{
  this->m_S.push_back( o );
  this->m_F.push_back( std::make_pair( a, TActivationFactory::New( a ) ) );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::FeedForwardNetwork< _S >::
TNatural ivqML::Model::FeedForwardNetwork< _S >::
number_of_layers( ) const
{
  if( this->m_S.size( ) > 0 )
    return( this->m_S.size( ) - 1 );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
init( )
{
  this->m_S.shrink_to_fit( );

  TNatural P = 0;
  this->m_C = this->m_S[ 0 ];
  for( TNatural l = 1; l < this->m_S.size( ); ++l )
  {
    TNatural i = this->m_S[ l - 1 ];
    TNatural o = this->m_S[ l ];
    P += o * ( i + 1 );
    this->m_C += ( this->m_S[ l ] << 1 );
  } // end for
  this->Superclass::set_number_of_parameters( P );
  this->random_fill( );

  // Create matrices and vectors
  this->m_W.clear( );
  this->m_B.clear( );
  TNatural s = 0;
  TScalar* b = this->begin( );
  for( TNatural l = 1; l < this->m_S.size( ); ++l )
  {
    TNatural i = this->m_S[ l - 1 ];
    TNatural o = this->m_S[ l ];

    this->m_W.push_back( TMap( b + s, i, o ) );
    this->m_B.push_back( TMap( b + s + ( i * o ), 1, o ) );

    s += o * ( i + 1 );
  } // end for
  this->m_W.shrink_to_fit( );
  this->m_B.shrink_to_fit( );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::FeedForwardNetwork< _S >::
TNatural ivqML::Model::FeedForwardNetwork< _S >::
cache_size( ) const
{
  return( TNatural( double( this->m_Cache.size( ) ) / double( this->m_C ) ) );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
resize_cache( const TNatural& s ) const
{
  if( s < this->cache_size( ) )
    return;

  this->Superclass::resize_cache( s * this->m_C );
  std::fill( this->m_Cache.begin( ), this->m_Cache.end( ), 0 );

  this->m_A.clear( );
  this->m_Z.clear( );

  TScalar* c = this->m_Cache.data( );
  this->m_A.push_back( TMap( c, s, this->m_S[ 0 ] ) );

  TNatural j = s * this->m_S[ 0 ];
  for( TNatural l = 1; l <= this->number_of_layers( ); ++l )
  {
    this->m_A.push_back( TMap( c + j, s, this->m_S[ l ] ) );
    j += s * this->m_S[ l ];

    this->m_Z.push_back( TMap( c + j, s, this->m_S[ l ] ) );
    j += s * this->m_S[ l ];
  } // end for
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
cost( TMatrix& G, const TMap& X, const TMap& Y, TScalar* J ) const
{
  // Forward propagation
  this->evaluate( X );

  // Check sizes
  TNatural n = this->number_of_parameters( );
  if( G.size( ) != n )
    G = TMatrix::Zero( 1, n );

  // Backwards propagation
  TScalar m = TScalar( X.rows( ) );
  TNatural L = this->number_of_layers( );
  TMatrix D = this->m_A[ L ] - Y;
  for( TNatural l = L; l > 0; --l )
  {
    // Update derivatives
    n -= D.cols( );
    TMap( G.data( ) + n, 1, D.cols( ) ) = D.colwise( ).mean( );
    n -= this->m_A[ l - 1 ].cols( ) * D.cols( );
    TMap( G.data( ) + n, this->m_A[ l - 1 ].cols( ), D.cols( )  )
      =
      ( this->m_A[ l - 1 ].transpose( ) * D ) / m;

    // Update delta if there is more back layers
    if( l > 1 )
    {
      D = ( D * this->m_W[ l - 1 ].transpose( ) ).eval( );
      TMatrix Zp( D.rows( ), D.cols( ) );
      TMap mZp(  Zp.data( ), Zp.rows( ), Zp.cols( ) );
      this->m_F[ l - 2 ].second( mZp, this->m_Z[ l - 2 ], true );
      D.array( ) *= Zp.array( );
    } // end if
  } // end for
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::FeedForwardNetwork< _S >::
TMap& ivqML::Model::FeedForwardNetwork< _S >::
_input_cache( ) const
{
  return( this->m_A[ 0 ] );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::FeedForwardNetwork< _S >::
TMap& ivqML::Model::FeedForwardNetwork< _S >::
_output_cache( ) const
{
  return( this->m_A.back( ) );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
_evaluate( const TNatural& m ) const
{
  TNatural L = this->number_of_layers( );
  for( TNatural l = 0; l < L; ++l )
  {
    this->m_Z[ l ] =
      ( this->m_A[ l ] * this->m_W[ l ] ).rowwise( )
      +
      this->m_B[ l ].row( 0 );
    this->m_F[ l ].second( this->m_A[ l + 1 ], this->m_Z[ l ], false );
  } // end for
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
_from_stream( std::istream& i )
{
  TNatural L, in, out;
  std::string a;

  i >> L >> in >> out >> a;
  this->add_layer( in, out, a );

  for( TNatural l = 1; l < L; ++l )
  {
    i >> out >> a;
    this->add_layer( out, a );
  } // end for

  this->init( );

  i >> a;
  std::transform(
    a.begin( ), a.end( ), a.begin( ),
    []( unsigned char c ){ return( std::tolower( c ) ); }
    );

  if( a == "random" )
  {
    // Do nothing since init() randomly fills
  }
  else if( a == "zeros" )
  {
    std::transform(
      this->begin( ), this->end( ), this->begin( ),
      []( const TScalar& v ){ return( TScalar( 0 ) ); }
      );
  }
  else if( a == "ones" )
  {
    std::transform(
      this->begin( ), this->end( ), this->begin( ),
      []( const TScalar& v ){ return( TScalar( 1 ) ); }
      );
  }
  else
  {
    TNatural P = std::atoi( a.c_str( ) );
    if( P != this->m_Parameters.size( ) )
      throw std::length_error( "Length mismatch while reading model." );
    for( TNatural p = 0; p < P; ++p )
      i >> *( this->begin( ) + p );
  } // end if
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
_to_stream( std::ostream& o ) const
{
  TNatural L = this->number_of_layers( );
  o << L << " " << this->m_S[ 0 ] << std::endl;
  for( TNatural l = 0; l < L; ++l )
    o
      << this->m_S[ l + 1 ] << " "
      << this->m_F[ l ].first << std::endl;
  this->Superclass::_to_stream( o );
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::FeedForwardNetwork< float >;
template class ivqML_EXPORT ivqML::Model::FeedForwardNetwork< double >;
template class ivqML_EXPORT ivqML::Model::FeedForwardNetwork< long double >;

// eof - $RCSfile$
