// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include <ivqML/Model/NeuralNetwork/FeedForward.h>

// -------------------------------------------------------------------------
template< class _S >
ivqML::Model::NeuralNetwork::FeedForward< _S >::
FeedForward( )
  : Superclass( TNatural( 0 ) )
{
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::NeuralNetwork::FeedForward< _S >::
random_fill( )
{
  this->Superclass::random_fill( );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::NeuralNetwork::FeedForward< _S >::
TNatural ivqML::Model::NeuralNetwork::FeedForward< _S >::
number_of_inputs( ) const
{
  if( this->m_S.size( ) > 0 )
    return( this->m_S[ 0 ] );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::NeuralNetwork::FeedForward< _S >::
set_number_of_inputs( const TNatural& p )
{
  // WARNING: do nothing!
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::NeuralNetwork::FeedForward< _S >::
TNatural ivqML::Model::NeuralNetwork::FeedForward< _S >::
number_of_outputs( ) const
{
  if( this->m_S.size( ) > 0 )
    return( this->m_S.back( ) );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::NeuralNetwork::FeedForward< _S >::
set_number_of_parameters( const TNatural& p )
{
  // WARNING: do nothing!
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::NeuralNetwork::FeedForward< _S >::
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
void ivqML::Model::NeuralNetwork::FeedForward< _S >::
add_layer( const TNatural& o, const std::string& a )
{
  this->m_S.push_back( o );
  this->m_F.push_back( std::make_pair( a, TActivationFactory::New( a ) ) );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::NeuralNetwork::FeedForward< _S >::
TNatural ivqML::Model::NeuralNetwork::FeedForward< _S >::
number_of_layers( ) const
{
  if( this->m_S.size( ) > 0 )
    return( this->m_S.size( ) - 1 );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::NeuralNetwork::FeedForward< _S >::
init( )
{
  this->m_S.shrink_to_fit( );

  TNatural P = 0;
  for( TNatural l = 1; l < this->m_S.size( ); ++l )
  {
    TNatural i = this->m_S[ l - 1 ];
    TNatural o = this->m_S[ l ];
    P += o * ( i + 1 );
  } // end for
  this->Superclass::set_number_of_parameters( P );
  this->random_fill( );

  // Create matrices and vectors
  this->m_W.clear( );
  this->m_B.clear( );
  this->m_BSize = this->m_S[ 0 ];
  TNatural s = 0;
  TScalar* b = this->m_Parameters;
  for( TNatural l = 1; l < this->m_S.size( ); ++l )
  {
    TNatural i = this->m_S[ l - 1 ];
    TNatural o = this->m_S[ l ];
    this->m_BSize += ( o << 1 );

    this->m_W.push_back( TMap( b + s, o, i ) );
    this->m_B.push_back( TMap( b + s + ( i * o ), o, 1 ) );

    s += o * ( i + 1 );
  } // end for
  this->m_W.shrink_to_fit( );
  this->m_B.shrink_to_fit( );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::NeuralNetwork::FeedForward< _S >::
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
      this->m_Parameters, this->m_Parameters + this->m_Size,
      this->m_Parameters,
      []( const TScalar& v ){ return( TScalar( 0 ) ); }
      );
  }
  else if( a == "ones" )
  {
    std::transform(
      this->m_Parameters, this->m_Parameters + this->m_Size,
      this->m_Parameters,
      []( const TScalar& v ){ return( TScalar( 1 ) ); }
      );
  }
  else
  {
    TNatural P = std::atoi( a.c_str( ) );
    if( P != this->m_Size )
      throw std::length_error( "Length mismatch while reading model." );
    for( TNatural p = 0; p < P; ++p )
      i >> this->m_Parameters[ p ];
  } // end if
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::NeuralNetwork::FeedForward< _S >::
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
template class ivqML_EXPORT
ivqML::Model::NeuralNetwork::FeedForward< float >;

template class ivqML_EXPORT
ivqML::Model::NeuralNetwork::FeedForward< double >;

template class ivqML_EXPORT
ivqML::Model::NeuralNetwork::FeedForward< long double >;

// eof - $RCSfile$
