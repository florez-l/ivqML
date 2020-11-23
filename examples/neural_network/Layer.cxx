// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "Layer.h"
#include <cassert>
#include <random>

// -------------------------------------------------------------------------
template< class _TScl >
Layer< _TScl >::
Layer( )
{
  this->m_W = TMatrix::Zero( 1, 1 );
  this->m_B = TColVector::Zero( 1 );
  this->m_S = nullptr;
}

// -------------------------------------------------------------------------
template< class _TScl >
Layer< _TScl >::
Layer( unsigned int i_size, unsigned int o_size, const std::string& f )
{
  this->m_W = TMatrix::Zero( o_size, i_size );
  this->m_B = TColVector::Zero( o_size );
  this->m_S = TFactory::get( )->create( f );
}

// -------------------------------------------------------------------------
template< class _TScl >
Layer< _TScl >::
Layer( const TMatrix& w, const TColVector& b, const std::string& f )
{
  assert( w.rows( ) == b.rows( ) && "Incompatible row count." );

  this->m_W = w;
  this->m_B = b;
  this->m_S = TFactory::get( )->create( f );
}

// -------------------------------------------------------------------------
template< class _TScl >
Layer< _TScl >::
Layer( const Self& o )
{
  this->m_W = o.m_W;
  this->m_B = o.m_B;
  this->m_S = ( o.m_S != nullptr )? o.m_S->copy( ): nullptr;
}

// -------------------------------------------------------------------------
template< class _TScl >
Layer< _TScl >::
~Layer( )
{
  if( this->m_S != nullptr )
    delete this->m_S;
}

// -------------------------------------------------------------------------
template< class _TScl >
typename Layer< _TScl >::
Self& Layer< _TScl >::
operator=( const Self& o )
{
  if( this->m_S != nullptr )
    delete this->m_S;
  this->m_W = o.m_W;
  this->m_B = o.m_B;
  this->m_S = ( o.m_S != nullptr )? o.m_S->copy( ): nullptr;
  return( *this );
}

// -------------------------------------------------------------------------
template< class _TScl >
unsigned int Layer< _TScl >::
input_size( ) const
{
  return( this->m_W.cols( ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
unsigned int Layer< _TScl >::
output_size( ) const
{
  return( this->m_W.rows( ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
void Layer< _TScl >::
init( )
{
  unsigned int r = this->m_W.rows( );
  unsigned int c = this->m_W.cols( );
  std::random_device rd;
  std::mt19937 gen( rd( ) );
  std::uniform_real_distribution< TScalar > dis( -1, 1 );

  this->m_W = TMatrix::Zero( r, c ).unaryExpr(
    [&]( TScalar not_used ) -> TScalar { return( dis( gen ) ); }
    );
  this->m_B = TColVector::Zero( r ).unaryExpr(
    [&]( TScalar not_used ) -> TScalar { return( 0 /*dis( gen )*/ ); }
    );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename Layer< _TScl >::
TScalar Layer< _TScl >::
regularization( ) const
{
  return( this->m_W.squaredNorm( ) + this->m_B.squaredNorm( ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename Layer< _TScl >::
TMatrix Layer< _TScl >::
f( const TMatrix& x ) const
{
  assert( this->m_S != nullptr && "No sigma function defined" );
  assert( this->m_W.cols( ) == x.rows( ) && "Invalid sizes" );

  return( this->m_S->f( ( this->m_W * x ).colwise( ) + this->m_B ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename Layer< _TScl >::
TMatrix Layer< _TScl >::
f( const TMatrix& x, TMatrix& z ) const
{
  assert( this->m_S != nullptr && "No sigma function defined" );
  assert( this->m_W.cols( ) == x.rows( ) && "Invalid sizes" );

  z = ( this->m_W * x ).colwise( ) + this->m_B;
  return(  this->m_S->f( z ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
void Layer< _TScl >::
_read_from( std::istream& i )
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
  this->m_S = TFactory::get( )->create( func_type );
}

// -------------------------------------------------------------------------
template< class _TScl >
void Layer< _TScl >::
_copy_to( std::ostream& o ) const
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
  o << std::endl << this->m_S->name( ) << std::endl;
}

// -------------------------------------------------------------------------
template class Layer< float >;
template class Layer< double >;
template class Layer< long double >;

// eof - $RCSfile$
