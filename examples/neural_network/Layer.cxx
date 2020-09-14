// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "Layer.h"
#include <cassert>

// -------------------------------------------------------------------------
template< class _TScalar >
Layer< _TScalar >::
Layer( unsigned int i_size, unsigned int o_size, const TActivation& f )
{
  this->m_W = TMatrix::Zero( o_size, i_size );
  this->m_B = TColVector::Zero( o_size );
  this->m_sigma = f;
}

// -------------------------------------------------------------------------
template< class _TScalar >
Layer< _TScalar >::
Layer( const TMatrix& w, const TColVector& b, const TActivation& f )
{
  assert( w.rows( ) == b.rows( ) );

  this->m_W = w;
  this->m_B = b;
  this->m_sigma = f;
}

// -------------------------------------------------------------------------
template< class _TScalar >
Layer< _TScalar >::
Layer( const Self& other )
{
  this->m_W = other.m_W;
  this->m_B = other.m_B;
  this->m_sigma = other.m_sigma;
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename Layer< _TScalar >::
Self& Layer< _TScalar >::
operator=( const Self& other )
{
  this->m_W = other.m_W;
  this->m_B = other.m_B;
  this->m_sigma = other.m_sigma;
  return( *this );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename Layer< _TScalar >::
TMatrix& Layer< _TScalar >::
W( )
{
  return( this->m_W );
}

// -------------------------------------------------------------------------
template< class _TScalar >
const typename Layer< _TScalar >::
TMatrix& Layer< _TScalar >::
W( ) const
{
  return( this->m_W );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename Layer< _TScalar >::
TColVector& Layer< _TScalar >::
B( )
{
  return( this->m_B );
}

// -------------------------------------------------------------------------
template< class _TScalar >
const typename Layer< _TScalar >::
TColVector& Layer< _TScalar >::
B( ) const
{
  return( this->m_B );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename Layer< _TScalar >::
TActivation& Layer< _TScalar >::
sigma( )
{
  return( this->m_sigma );
}

// -------------------------------------------------------------------------
template< class _TScalar >
const typename Layer< _TScalar >::
TActivation& Layer< _TScalar >::
sigma( ) const
{
  return( this->m_sigma );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void Layer< _TScalar >::
init( bool randomly )
{
  unsigned int r = this->m_W.rows( );
  unsigned int c = this->m_W.cols( );
  if( randomly )
  {
    this->m_W = TMatrix::Random( r, c );
    this->m_B = TColVector::Random( r );
  }
  else
  {
    this->m_W = TMatrix::Zero( r, c );
    this->m_B = TColVector::Zero( r );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename Layer< _TScalar >::
TColVector Layer< _TScalar >::
operator()( const TColVector& z ) const
{
  return( this->m_sigma( ( this->m_W * z ) + this->m_B ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void Layer< _TScalar >::
_ReadFrom( std::istream& i )
{
  unsigned int R, C;
  i >> R >> C;
  this->m_W = TMatrix::Zero( R, C );
  this->m_B = TColVector::Zero( R );
  for( unsigned int r = 0; r < R; ++r )
    for( unsigned int c = 0; c < C; ++c )
      i >> this->m_W( r, c );
  for( unsigned int r = 0; r < R; ++r )
    i >> this->m_B( r, 0 );
  std::string func_type;
  i >> func_type;
}

// -------------------------------------------------------------------------
template< class _TScalar >
void Layer< _TScalar >::
_CopyTo( std::ostream& o ) const
{
  o << this->m_W.rows( ) << " " << this->m_W.cols( ) << std::endl;
  for( unsigned int r = 0; r < this->m_W.rows( ); ++r )
  {
    for( unsigned int c = 0; c < this->m_W.cols( ); ++c )
      o << this->m_W( r, c ) << " ";
    o << std::endl;
  } // end for
  for( unsigned int r = 0; r < this->m_B.rows( ); ++r )
    o << this->m_B( r, 0 ) << " ";
  o << std::endl << typeid( this->m_sigma ).name( ) << std::endl;
}

// -------------------------------------------------------------------------
template class Layer< float >;
template class Layer< double >;
template class Layer< long double >;

// eof - $RCSfile$
