// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "NeuralNetwork.h"

#include <cassert>
#include <map>
#include <regex>
#include <sstream>
#include <boost/algorithm/string.hpp>

// -------------------------------------------------------------------------
template< class _TScl >
NeuralNetwork< _TScl >::
NeuralNetwork( )
{
}

// -------------------------------------------------------------------------
template< class _TScl >
NeuralNetwork< _TScl >::
NeuralNetwork( const Self& o )
{
  this->m_L.clear( );
  this->m_L.insert( this->m_L.begin( ), o.m_L.begin( ), o.m_L.end( ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename NeuralNetwork< _TScl >::
Self& NeuralNetwork< _TScl >::
operator=( const Self& o )
{
  this->m_L.clear( );
  this->m_L.insert( this->m_L.begin( ), o.m_L.begin( ), o.m_L.end( ) );
  return( *this );
}

// -------------------------------------------------------------------------
template< class _TScl >
void NeuralNetwork< _TScl >::
add( unsigned int i, unsigned int o, const std::string& f )
{
  this->add( TLayer( i, o, f ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
void NeuralNetwork< _TScl >::
add( unsigned int o, const std::string& f )
{
  assert( this->m_L.size( ) > 0 && "At least one layer is needed" );

  this->add( this->m_L.back( ).output_size( ), o, f );
}

// -------------------------------------------------------------------------
template< class _TScl >
void NeuralNetwork< _TScl >::
add( const TMatrix& w, const TColVector& b, const std::string& f )
{
  this->add( TLayer( w, b, f ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
void NeuralNetwork< _TScl >::
add( const TLayer& l )
{
  if( this->m_L.size( ) > 0 )
    assert(
      l.input_size( ) == this->m_L.back( ).output_size( ) && "Invalid sizes"
      );
  this->m_L.push_back( l );

  // Normalization transforms
  if( this->m_L.size( ) == 1 )
  {
    unsigned int n = this->m_L[ 0 ].input_size( );
    this->m_NormalizationOffset = TColVector::Zero( n );
    this->m_NormalizationScale = TMatrix::Identity( n, n );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScl >
void NeuralNetwork< _TScl >::
load_topology( std::istream& is )
{
  unsigned int L;
  std::map< unsigned int, unsigned long > K;
  std::map< unsigned int, std::string > F;

  // Read parameters
  std::regex r( "\\s+" );
  std::string line;
  while( std::getline( is, line ) )
  {
    line = std::regex_replace( line, r, "" );
    std::vector< std::string > tokens;
    boost::algorithm::split( tokens, line, boost::is_any_of( "=" ) );
    if( tokens.size( ) == 2 )
    {
      if( tokens[ 0 ] == "L" )
      {
        std::istringstream data( tokens[ 1 ] );
        data >> L;
      }
      else
      {
        if( tokens[ 0 ][ 0 ] == 'k' )
        {
          std::istringstream i_str( tokens[ 0 ].substr( 1 ) );
          std::istringstream k_str( tokens[ 1 ] );
          unsigned int i;
          unsigned long k;
          i_str >> i;
          k_str >> k;
          K[ i ] = k;
        }
        else if( tokens[ 0 ][ 0 ] == 'f' )
        {
          std::istringstream i_str( tokens[ 0 ].substr( 1 ) );
          unsigned int i;
          i_str >> i;
          F[ i ] = tokens[ 1 ];
        } // end if
      } // end if
    } // end if
  } // end while

  // Real build
  if( F.size( ) == L && K.size( ) == L + 1 )
    for( unsigned int l = 0; l < L; ++l )
      this->add( K[ l ], K[ l + 1 ], F[ l ] );
}

// -------------------------------------------------------------------------
template< class _TScl >
void NeuralNetwork< _TScl >::
set( unsigned int l, const TMatrix& w, const TColVector& b )
{
  if( l < this->m_L.size( ) )
  {
    this->m_L[ l ].weights( ) = w;
    this->m_L[ l ].biases( ) = b;
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScl >
unsigned int NeuralNetwork< _TScl >::
number_of_layers( ) const
{
  return( this->m_L.size( ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename NeuralNetwork< _TScl >::
TMatrix& NeuralNetwork< _TScl >::
weights( unsigned int l )
{
  static TMatrix w( 1, 1 );
  if( l < this->m_L.size( ) )
    return( this->m_L[ l ].weights( ) );
  else
    return( w );
}

// -------------------------------------------------------------------------
template< class _TScl >
const typename NeuralNetwork< _TScl >::
TMatrix& NeuralNetwork< _TScl >::
weights( unsigned int l ) const
{
  static const TMatrix w( 1, 1 );
  if( l < this->m_L.size( ) )
    return( this->m_L[ l ].weights( ) );
  else
    return( w );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename NeuralNetwork< _TScl >::
TColVector& NeuralNetwork< _TScl >::
biases( unsigned int l )
{
  static TColVector b( 1 );
  if( l < this->m_L.size( ) )
    return( this->m_L[ l ].biases( ) );
  else
    return( b );
}

// -------------------------------------------------------------------------
template< class _TScl >
const typename NeuralNetwork< _TScl >::
TColVector& NeuralNetwork< _TScl >::
biases( unsigned int l ) const
{
  static const TColVector b( 1 );
  if( l < this->m_L.size( ) )
    return( this->m_L[ l ].biases( ) );
  else
    return( b );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename NeuralNetwork< _TScl >::
TActivation* NeuralNetwork< _TScl >::
sigma( unsigned int l )
{
  if( l < this->m_L.size( ) )
    return( this->m_L[ l ].sigma( ) );
  else
    return( nullptr );
}

// -------------------------------------------------------------------------
template< class _TScl >
const typename NeuralNetwork< _TScl >::
TActivation* NeuralNetwork< _TScl >::
sigma( unsigned int l ) const
{
  if( l < this->m_L.size( ) )
    return( this->m_L[ l ].sigma( ) );
  else
    return( nullptr );
}

// -------------------------------------------------------------------------
template< class _TScl >
void NeuralNetwork< _TScl >::
setNormalizationOffset( const TColVector& o )
{
  this->m_NormalizationOffset = o;
}

// -------------------------------------------------------------------------
template< class _TScl >
void NeuralNetwork< _TScl >::
setNormalizationScale( const TMatrix& s )
{
  this->m_NormalizationScale = s;
}

// -------------------------------------------------------------------------
template< class _TScl >
void NeuralNetwork< _TScl >::
init( )
{
  // Weights
  for( TLayer& l: this->m_L )
    l.init( );

  // Normalization transforms
  unsigned int n = this->m_L[ 0 ].input_size( );
  this->m_NormalizationOffset = TColVector::Zero( n );
  this->m_NormalizationScale = TMatrix::Identity( n, n );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename NeuralNetwork< _TScl >::
TMatrix NeuralNetwork< _TScl >::
f( const TMatrix& x ) const
{
  typename TLayers::const_iterator lIt = this->m_L.begin( );
  TMatrix z = lIt->f(
    this->m_NormalizationScale *
    ( x.colwise( ) + this->m_NormalizationOffset )
    );
  for( lIt++; lIt != this->m_L.end( ); ++lIt )
    z = lIt->f( z );
  return( z );
}

// -------------------------------------------------------------------------
template< class _TScl >
void NeuralNetwork< _TScl >::
f( std::vector< TMatrix >& a, std::vector< TMatrix >& z ) const
{
  typename TLayers::const_iterator lIt = this->m_L.begin( );
  typename std::vector< TMatrix >::iterator aIt, bIt, zIt;
  aIt = bIt = a.begin( );
  zIt = z.begin( );

  for( bIt++; lIt != this->m_L.end( ); ++lIt, ++aIt, ++bIt, ++zIt )
    *bIt = lIt->f( *aIt, *zIt );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename NeuralNetwork< _TScl >::
TMatrix NeuralNetwork< _TScl >::
t( const TMatrix& x ) const
{
  return( this->m_L.back( ).sigma( )->t( this->f( x ) ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename NeuralNetwork< _TScl >::
TMatrix NeuralNetwork< _TScl >::
_d( const unsigned int& l, const TMatrix& z ) const
{
  assert( l < this->m_L.size( ) && "Layer does not exist" );

  return( this->m_L[ l ].sigma( )->d( z ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
void NeuralNetwork< _TScl >::
_read_from( std::istream& i )
{
  unsigned int L;
  i >> L;
  for( unsigned int n = 0; n < L; ++n )
  {
    TLayer l;
    i >> l;
    this->add( l );
  } // end for

  // Read normalization parameters
  unsigned int N = this->m_L[ 0 ].input_size( );
  this->m_NormalizationOffset = TColVector::Zero( N );
  this->m_NormalizationScale = TMatrix::Identity( N, N );
  for( unsigned int n = 0; n < N; ++n )
    i >> this->m_NormalizationOffset( n, 0 );
  for( unsigned int x = 0; x < N; ++x )
    for( unsigned int y = 0; y < N; ++y )
      i >> this->m_NormalizationScale( x, y );
}

// -------------------------------------------------------------------------
template< class _TScl >
void NeuralNetwork< _TScl >::
_copy_to( std::ostream& o ) const
{
  o << this->m_L.size( ) << std::endl;
  for( const TLayer& l: this->m_L )
    o << l << std::endl;
  o << this->m_NormalizationOffset << std::endl;
  o << this->m_NormalizationScale << std::endl;
}

// -------------------------------------------------------------------------
template class NeuralNetwork< float >;
template class NeuralNetwork< double >;
template class NeuralNetwork< long double >;

// eof - $RCSfile$
