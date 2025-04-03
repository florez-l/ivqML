// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/NeuralNetwork/FeedForward.h>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
FeedForward( )
  : Superclass( 0 )
{
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
~FeedForward( )
{
  this->m_FwdBuf.free( );
  this->m_BwdBuf.free( );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
set_size( const TNatural& n )
{
  // Do nothing
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
set_input_size( const TNatural& n0 )
{
  this->m_N.clear( );
  this->m_W.clear( );
  this->m_B.clear( );
  this->m_A.clear( );

  this->m_N.push_back( n0 );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
typename ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TNatural ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
input_size( ) const
{
  if( this->m_N.size( ) > 0 )
    return( this->m_N[ 0 ] );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
typename ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TNatural ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
output_size( ) const
{
  if( this->m_N.size( ) > 0 )
    return( this->m_N.back( ) );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
add_layer( const TNatural& n, const std::string& a )
{
  this->m_N.push_back( n );
  this->m_A.push_back( TFunctions::Get( a ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
typename ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TNatural ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
number_of_layers( ) const
{
  if( this->m_N.size( ) > 0 )
    return( this->m_N.size( ) - 1 );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
init( std::function< TReal( ) > g )
{
  this->m_N.shrink_to_fit( );
  this->m_W.clear( );
  this->m_B.clear( );

  this->m_S = 0;
  for( TNatural n = 0; n < this->m_N.size( ); n++ )
    this->m_S += ( this->m_N[ n - 1 ] + 1 ) * this->m_N[ n ];

  this->Superclass::init( g );

  TReal* p = this->m_P;
  for( TNatural l = 1; l < this->m_N.size( ); ++l )
  {
    this->m_W.push_back( TMatrixMap( p, this->m_N[ l - 1 ], this->m_N[ l ] ) );
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
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::SBuffer::
allocate( const std::vector< TNatural >& n, const TNatural& m, bool keepAZ )
{
  if( this->M < m )
  {
    this->free( );
    if( keepAZ )
    {
      this->N = std::accumulate( n.begin( ), n.end( ), 0 );
      this->B
        =
        reinterpret_cast< TReal* >(
          std::calloc( ( this->N * m ) << 1, sizeof( TReal ) )
          );
    }
    else
    {
      this->N = *( std::max_element( n.begin( ), n.end( ) ) );
      this->B
        =
        reinterpret_cast< TReal* >(
          std::calloc( this->N * m, sizeof( TReal ) )
          );
    } // end if
  } // end if

  if( this->M != m && this->B != nullptr )
  {
    TReal* a = this->B;
    TReal* z = this->B + ( ( keepAZ )? ( ( this->N * m ) + n[ 0 ] ): 0 );
    this->A.clear( );
    this->Z.clear( );

    this->A.push_back( TMatrixMap( a, n[ 0 ], m ) );
    if( keepAZ ) a += this->A.back( ).size( );
    for( TNatural l = 1; l < n.size( ); ++l )
    {
      this->A.push_back( TMatrixMap( a, n[ l ], m ) );
      this->Z.push_back( TMatrixMap( z, n[ l ], m ) );
      if( keepAZ ) a += this->A.back( ).size( );
      if( keepAZ ) z += this->Z.back( ).size( );
    } // end for
  } // end if
  this->M = m;
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::SBuffer::
free( )
{
  this->M = 0;
  this->Z.clear( );
  this->A.clear( );
  if( this->B != nullptr )
    std::free( this->B );
  this->B = nullptr;
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
_eval( SBuffer& b ) const
{
  TNatural L = this->number_of_layers( );
  for( TNatural l = 0; l < L; ++l )
  {
    b.Z[ l ] = ( this->m_W[ l ] * b.A[ l ] ).colwise( ) + this->m_B[ l ];
    this->m_A[ l ].second( b.A[ l + 1 ], b.Z[ l ], false );
  } // end for
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
_to_stream( std::ostream& o ) const
{
  this->Superclass::_to_stream( o );
}

// -------------------------------------------------------------------------
namespace ivqML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      template class ivqML_EXPORT FeedForward< float, unsigned int >;
      template class ivqML_EXPORT FeedForward< float, unsigned long >;
      template class ivqML_EXPORT FeedForward< float, unsigned long long >;
      template class ivqML_EXPORT FeedForward< double, unsigned int >;
      template class ivqML_EXPORT FeedForward< double, unsigned long >;
      template class ivqML_EXPORT FeedForward< double, unsigned long long >;
      template class ivqML_EXPORT FeedForward< long double, unsigned int >;
      template class ivqML_EXPORT FeedForward< long double, unsigned long >;
      template class ivqML_EXPORT FeedForward< long double, unsigned long long >;
    } // end namespace
  } // end namespace
} // end namespace

// eof - $RCSfile$
