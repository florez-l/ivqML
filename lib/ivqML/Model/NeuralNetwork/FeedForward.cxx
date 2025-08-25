// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/NeuralNetwork/FeedForward.h>

// -------------------------------------------------------------------------
template< class _TReal >
ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
FeedForward( )
  : Superclass( 0 )
{
}

// -------------------------------------------------------------------------
template< class _TReal >
ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
~FeedForward( )
{
  this->free_fitting_buffer( );
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
set_size( const TNatural& n )
{
  // Do nothing
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
set_input_size( const TNatural& n0 )
{
  this->m_N.clear( );
  this->m_W.clear( );
  this->m_B.clear( );
  this->m_A.clear( );

  this->m_N.push_back( n0 );
}

// -------------------------------------------------------------------------
template< class _TReal >
typename ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
TNatural ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
input_size( ) const
{
  if( this->m_N.size( ) > 0 )
    return( this->m_N[ 0 ] );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _TReal >
typename ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
TNatural ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
output_size( ) const
{
  if( this->m_N.size( ) > 0 )
    return( this->m_N.back( ) );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
add_layer( const TNatural& n, const std::string& a )
{
  this->m_N.push_back( n );
  this->m_A.push_back( TFunctions::Get( a ) );
}

// -------------------------------------------------------------------------
template< class _TReal >
typename ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
TNatural ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
number_of_layers( ) const
{
  if( this->m_N.size( ) > 0 )
    return( this->m_N.size( ) - 1 );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
init( std::function< TReal( ) > g )
{
  this->m_N.shrink_to_fit( );
  this->m_W.clear( );
  this->m_B.clear( );

  this->m_S = 0;
  for( TNatural n = 1; n < this->m_N.size( ); n++ )
    this->m_S += ( this->m_N[ n - 1 ] + 1 ) * this->m_N[ n ];

  this->Superclass::init( g );

  TReal* p = this->m_P;
  for( TNatural l = 1; l < this->m_N.size( ); ++l )
  {
    this->m_W.push_back( TMatrixMap( p, this->m_N[ l ], this->m_N[ l - 1 ] ) );
    p += this->m_W.back( ).size( );
    this->m_B.push_back( TColumnMap( p, this->m_N[ l ], 1 ) );
    p += this->m_B.back( ).size( );
  } // end for

  // Configure cost
  if( this->m_A.back( ).first == "sigmoid" )
    this->m_J.set_type_to_MCE( );
  else if( this->m_A.back( ).first == "softmax" )
    this->m_J.set_type_to_CCE( );
  else
    this->m_J.set_type_to_MSE( );
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
allocate_fitting_buffer( const TNatural& M ) const
{
  auto bN = this->m_N.begin( );
  auto eN = this->m_N.end( );

  TNatural N = *( bN++ );
  N += std::accumulate( bN, eN, 0 ) << 1;
  N *= M;

  this->free_fitting_buffer( );
  this->m_FittingBuffer
    =
    reinterpret_cast< TReal* >( std::calloc( N, sizeof( TReal ) ) );
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
free_fitting_buffer( ) const
{
  if( this->m_FittingBuffer != nullptr )
    std::free( this->m_FittingBuffer );
  this->m_FittingBuffer = nullptr;
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
_eval( TReal* Ab, TReal* Zb, const TNatural& M, bool offset ) const
{
  TReal* A = Ab;
  TReal* Z = Zb;

  TNatural L = this->number_of_layers( );
  for( TNatural l = 0; l < L; ++l )
  {
    TNatural i = this->m_N[ l ];
    TNatural o = this->m_N[ l + 1 ];

    TMatrixMap( Z, o, M )
      =
      ( this->m_W[ l ] * TMatrixMap( A, i, M ) ).colwise( )
      +
      this->m_B[ l ];
    if( offset )
      A += i * M;

    this->m_A[ l ].second(
      TMatrixMap( A, o, M ),
      TMatrixMap( Z, o, M ),
      false
      );
    if( offset )
      Z += o * M;
  } // end for
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal >::
_to_stream( std::ostream& o ) const
{
  // TODO: this->Superclass::_to_stream( o );
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
