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
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
add_layer( const TNatural& n, const std::string& a )
{
  this->m_N.push_back( n );
  this->m_A.push_back( TFunctions::Get( a ) );
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
  this->m_FwdBuf.N = 0;
  this->m_BwdBuf.N = 0;
  for( TNatural n = 1; n < this->m_N.size( ); n++ )
  {
    this->m_S += ( this->m_N[ n - 1 ] + 1 ) * this->m_N[ n ];

    this->m_FwdBuf.N
      =
      ( this->m_FwdBuf.N < this->m_N[ n ] )? this->m_N[ n ]: this->m_FwdBuf.N;
    this->m_BwdBuf.N += this->m_N[ n ];
  } // end for

  this->Superclass::init( g );

  TReal* p = this->m_P;
  for( TNatural l = 1; l < this->m_N.size( ); ++l )
  {
    this->m_W.push_back( TMatrixMap( p, this->m_N[ l - 1 ], this->m_N[ l ] ) );
    p += this->m_W.back( ).size( );
    this->m_B.push_back( TColumnMap( p, this->m_N[ l ], 1 ) );
    p += this->m_B.back( ).size( );
  } // end for
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::SBuffer::
allocate( const TNatural& m, bool keepAZ )
{
  if( this->M != m )
  {
    this->free( );
    this->M = m;
    this->B
      =
      reinterpret_cast< TReal* >(
        std::calloc( ( this->N * this->M ) << 1, sizeof( TReal ) )
        );
    this->Z = this->B;
    this->A = this->B + ( this->N * this->M );
    this->K = keepAZ;
  } // end if
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::SBuffer::
free( )
{
  this->M = 0;
  this->Z = nullptr;
  this->A = nullptr;
  if( this->B != nullptr )
    std::free( this->B );
  this->B = nullptr;
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
_to_stream( std::ostream& o ) const
{
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
