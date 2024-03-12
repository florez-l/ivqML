// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include <ivqML/Model/NeuralNetwork/FeedForward.h>

// -------------------------------------------------------------------------
template< class _TScalar >
ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
FeedForward( )
  : Superclass( TNatural( 0 ) )
{
}

// -------------------------------------------------------------------------
template< class _TScalar >
bool ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
has_backpropagation( ) const
{
  return( true );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
TNatural ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
number_of_inputs( ) const
{
  if( this->m_S.size( ) > 0 )
    return( this->m_S[ 0 ] );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
set_number_of_inputs( const TNatural& p )
{
  this->m_S.clear( );
  this->m_S.push_back( p );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
TNatural ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
number_of_outputs( ) const
{
  if( this->m_S.size( ) > 0 )
    return( this->m_S.back( ) );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
TNatural ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
buffer_size( ) const
{
  TNatural s = this->m_S[ 0 ];
  for( TNatural i = 1; i < this->m_S.size( ); ++i )
    s += ( this->m_S[ i ] << 1 );
  return( s );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
set_number_of_parameters( const TNatural& p )
{
  // WARNING: do nothing!
}

// -------------------------------------------------------------------------
template< class _TScalar >
void ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
add_layer( const TNatural& i )
{
  this->set_number_of_inputs( i );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
add_layer( const TNatural& o, const std::string& a )
{
  this->m_S.push_back( o );
  this->m_A.push_back( std::make_pair( a, TActivationFactory::New( a ) ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
TNatural ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
number_of_layers( ) const
{
  if( this->m_S.size( ) > 0 )
    return( this->m_S.size( ) - 1 );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
init( )
{
  this->m_S.shrink_to_fit( );
  this->m_O.clear( );

  TNatural P = 0;
  for( TNatural l = 1; l < this->m_S.size( ); ++l )
  {
    TNatural i = this->m_S[ l - 1 ];
    TNatural o = this->m_S[ l ];

    this->m_O.push_back( P );
    this->m_O.push_back( P + ( o * i ) );

    P += o * ( i + 1 );
  } // end for
  this->m_O.shrink_to_fit( );

  this->Superclass::set_number_of_parameters( P );
  this->random_fill( );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
TMatMap ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
W( const TNatural& l )
{
  if( 1 <= l && l < this->m_S.size( ) )
    return(
      this->matrix(
        this->m_S[ l ], this->m_S[ l - 1 ],
        this->m_O[ ( l - 1 ) >> 1 ]
        )
      );
  else
    return( TMatMap( nullptr, 0, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
TMatCMap ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
W( const TNatural& l ) const
{
  if( 1 <= l && l < this->m_S.size( ) )
    return(
      this->matrix(
        this->m_S[ l ], this->m_S[ l - 1 ],
        this->m_O[ ( l - 1 ) >> 1 ]
        )
      );
  else
    return( TMatCMap( nullptr, 0, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
TColMap ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
B( const TNatural& l )
{
  if( 1 <= l && l < this->m_S.size( ) )
    return(
      this->column(
        this->m_S[ l ], this->m_O[ ( ( l - 1 ) >> 1 ) + 1 ]
        )
      );
  else
    return( TColMap( nullptr, 0, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
TColCMap ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
B( const TNatural& l ) const
{
  if( 1 <= l && l < this->m_S.size( ) )
    return(
      this->column(
        this->m_S[ l ], this->m_O[ ( ( l - 1 ) >> 1 ) + 1 ]
        )
      );
  else
    return( TColCMap( nullptr, 0, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
_from_stream( std::istream& i )
{
  TNatural L, in, out;
  std::string a;

  i >> L >> in >> out >> a;
  this->add_layer( in );
  this->add_layer( out, a );
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
      this->m_P.data( ), this->m_P.data( ) + this->m_P.size( ),
      this->m_P.data( ),
      []( const TScalar& v ){ return( TScalar( 0 ) ); }
      );
  }
  else if( a == "ones" )
  {
    std::transform(
      this->m_P.data( ), this->m_P.data( ) + this->m_P.size( ),
      this->m_P.data( ),
      []( const TScalar& v ){ return( TScalar( 1 ) ); }
      );
  }
  else
  {
    TNatural P = std::atoi( a.c_str( ) );
    if( P != this->m_P.size( ) )
      throw std::length_error( "Length mismatch while reading model." );
    for( TNatural p = 0; p < P; ++p )
      i >> this->m_P( p );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScalar >
void ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
_to_stream( std::ostream& o ) const
{
  TNatural L = this->number_of_layers( );
  o << L << " " << this->m_S[ 0 ] << std::endl;
  for( TNatural l = 0; l < L; ++l )
    o
      << this->m_S[ l + 1 ] << " " << this->m_A[ l ].first
      << std::endl;
  this->Superclass::_to_stream( o );
}

// -------------------------------------------------------------------------
namespace ivqML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      template class ivqML_EXPORT FeedForward< float >;
      template class ivqML_EXPORT FeedForward< double >;
      template class ivqML_EXPORT FeedForward< long double >;
    } // end namespace
  } // end namespace
} // end namespace

// eof - $RCSfile$
