// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Base.h>
#include <random>

// -------------------------------------------------------------------------
template< class _S >
ivqML::Model::Base< _S >::
Base( const TNatural& n )
{
  this->set_number_of_parameters( n );
}

// -------------------------------------------------------------------------
template< class _S >
ivqML::Model::Base< _S >::
~Base( )
{
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Base< _S >::
random_fill( )
{
  std::random_device r;
  std::mt19937 g( r( ) );
  std::uniform_real_distribution< _S > d( 0, 1 );
  for( TNatural i = 0; i < this->m_Parameters.size( ); ++i )
    this->m_Parameters[ i ] = d( g );
}

// -------------------------------------------------------------------------
template< class _S >
_S& ivqML::Model::Base< _S >::
operator[]( const TNatural& i )
{
  static _S zero = 0;
  if( i < this->m_Parameters.size( ) )
    return( this->m_Parameters[ i ] );
  else
  {
    zero = 0;
    return( zero );
  } // end if
}

// -------------------------------------------------------------------------
template< class _S >
const _S& ivqML::Model::Base< _S >::
operator[]( const TNatural& i ) const
{
  static const _S zero = 0;
  if( i < this->m_Parameters.size( ) )
    return( this->m_Parameters[ i ] );
  else
    return( zero );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::Base< _S >::
TNatural ivqML::Model::Base< _S >::
number_of_parameters( ) const
{
  return( this->m_Parameters.size( ) );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Base< _S >::
set_number_of_parameters( const TNatural& p )
{
  if( this->m_Parameters.size( ) != p )
  {
    if( p > 0 )
      this->m_Parameters.resize( p );
    else
      this->m_Parameters.clear( );
    this->m_Parameters.shrink_to_fit( );
  } // end if
  std::fill( this->m_Parameters.begin( ), this->m_Parameters.end( ), TScalar( 0 ) );
}

// -------------------------------------------------------------------------
template< class _S >
_S* ivqML::Model::Base< _S >::
begin( )
{
  return( this->m_Parameters.data( ) );
}

// -------------------------------------------------------------------------
template< class _S >
const _S* ivqML::Model::Base< _S >::
begin( ) const
{
  return( this->m_Parameters.data( ) );
}

// -------------------------------------------------------------------------
template< class _S >
_S* ivqML::Model::Base< _S >::
end( )
{
  return( this->m_Parameters.data( ) + this->m_Parameters.size( ) );
}

// -------------------------------------------------------------------------
template< class _S >
const _S* ivqML::Model::Base< _S >::
end( ) const
{
  return( this->m_Parameters.data( ) + this->m_Parameters.size( ) );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::Base< _S >::
TNatural ivqML::Model::Base< _S >::
_cache_size( ) const
{
  return( this->m_Cache.size( ) );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Base< _S >::
_resize_cache( const TNatural& s ) const
{
  if( s == 0 )
    this->m_Cache.clear( );
  if( s > this->m_Cache.size( ) )
    this->m_Cache.resize( s );
  this->m_Cache.shrink_to_fit( );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Base< _S >::
_from_stream( std::istream& i )
{
  // TODO
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Base< _S >::
_to_stream( std::ostream& o ) const
{
  o << this->m_Parameters.size( );
  for( TNatural i = 0; i < this->m_Parameters.size( ); ++i )
    o << " " << this->m_Parameters[ i ];
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Base< float >;
template class ivqML_EXPORT ivqML::Model::Base< double >;
template class ivqML_EXPORT ivqML::Model::Base< long double >;

// eof - $RCSfile$
