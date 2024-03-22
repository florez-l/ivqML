// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Base.h>
#include <cstdlib>
#include <random>

// -------------------------------------------------------------------------
template< class _TScl >
ivqML::Model::Base< _TScl >::
Base( const TNat& n )
{
  this->set_number_of_parameters( n );
}

// -------------------------------------------------------------------------
template< class _TScl >
ivqML::Model::Base< _TScl >::
~Base( )
{
  if( this->m_P != nullptr )
    std::free( this->m_P );
}

// -------------------------------------------------------------------------
template< class _TScl >
void ivqML::Model::Base< _TScl >::
random_fill( )
{
  if( this->m_S > 0 )
  {
    std::random_device r;
    std::mt19937 g( r( ) );
    std::uniform_real_distribution< TScl > d( -1, 1 );
    for( TNat i = 0; i < this->m_S; ++i )
      this->m_P[ i ] = d( g );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScl >
_TScl& ivqML::Model::Base< _TScl >::
operator[]( const TNat& i )
{
  static TScl zero = TScl( 0 );
  if( i < this->m_S )
    return( this->m_P[ i ] );
  else
  {
    zero = TScl( 0 );
    return( zero );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScl >
const _TScl& ivqML::Model::Base< _TScl >::
operator[]( const TNat& i ) const
{
  static const TScl zero = TScl( 0 );
  if( i < this->m_S )
    return( this->m_P[ i ] );
  else
    return( zero );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ivqML::Model::Base< _TScl >::
TNat ivqML::Model::Base< _TScl >::
buffer_size( ) const
{
  return( this->number_of_parameters( ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ivqML::Model::Base< _TScl >::
TNat ivqML::Model::Base< _TScl >::
number_of_parameters( ) const
{
  return( this->m_S );
}

// -------------------------------------------------------------------------
template< class _TScl >
void ivqML::Model::Base< _TScl >::
set_number_of_parameters( const TNat& p )
{
  if( this->m_S != p )
  {
    if( this->m_P != nullptr )
      std::free( this->m_P );
    this->m_S = p;
    if( this->m_S > 0 )
      this->m_P =
        reinterpret_cast< TScl* >( std::calloc( this->m_S, sizeof( TScl ) ) );
    else
      this->m_P = nullptr;
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ivqML::Model::Base< _TScl >::
TMatMap ivqML::Model::Base< _TScl >::
matrix( const TNat& r, const TNat& c, const TNat& o )
{
  if( ( r * c ) <= ( this->m_S - o ) )
    return( TMatMap( this->m_P + o, r, c ) );
  return( TMatMap( nullptr, 0, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ivqML::Model::Base< _TScl >::
TMatCMap ivqML::Model::Base< _TScl >::
matrix( const TNat& r, const TNat& c, const TNat& o ) const
{
  if( ( r * c ) <= ( this->m_S - o ) )
    return( TMatCMap( this->m_P + o, r, c ) );
  return( TMatCMap( nullptr, 0, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ivqML::Model::Base< _TScl >::
TColMap ivqML::Model::Base< _TScl >::
column( const TNat& r, const TNat& o )
{
  if( r <= ( this->m_S - o ) )
    return( TColMap( this->m_P + o, r, 1 ) );
  return( TColMap( nullptr, 0, 1 ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ivqML::Model::Base< _TScl >::
TColCMap ivqML::Model::Base< _TScl >::
column( const TNat& r, const TNat& o ) const
{
  if( r <= ( this->m_S - o ) )
    return( TColCMap( this->m_P + o, r, 1 ) );
  return( TColCMap( nullptr, 0, 1 ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ivqML::Model::Base< _TScl >::
TRowMap ivqML::Model::Base< _TScl >::
row( const TNat& c, const TNat& o )
{
  if( c <= ( this->m_S - o ) )
    return( TRowMap( this->m_P + o, 1, c ) );
  return( TRowMap( nullptr, 1, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ivqML::Model::Base< _TScl >::
TRowCMap ivqML::Model::Base< _TScl >::
row( const TNat& c, const TNat& o ) const
{
  if( c <= ( this->m_S - o ) )
    return( TRowCMap( this->m_P + o, 1, c ) );
  return( TRowCMap( nullptr, 1, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
bool ivqML::Model::Base< _TScl >::
has_backpropagation( ) const
{
  return( false );
}

// -------------------------------------------------------------------------
template< class _TScl >
void ivqML::Model::Base< _TScl >::
_from_stream( std::istream& i )
{
  TNat p;
  i >> p;
  this->set_number_of_parameters( p );
  for( TNat j = 0; j < this->m_S; ++j )
    i >> this->m_P[ j ];
}

// -------------------------------------------------------------------------
template< class _TScl >
void ivqML::Model::Base< _TScl >::
_to_stream( std::ostream& o ) const
{
  o << this->m_S;
  for( TNat i = 0; i < this->m_S; ++i )
    o << " " << this->m_P[ i ];
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Base< float >;
template class ivqML_EXPORT ivqML::Model::Base< double >;
template class ivqML_EXPORT ivqML::Model::Base< long double >;

// eof - $RCSfile$
