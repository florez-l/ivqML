// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Base.h>
#include <cstdlib>
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
  if( this->m_Parameters != nullptr )
  {
    std::free( this->m_Parameters );
    this->m_Parameters = nullptr;
  } // end if
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Base< _S >::
random_fill( )
{
  std::random_device r;
  std::mt19937 g( r( ) );
  std::uniform_real_distribution< _S > d( 0, 1 );
  for( TNatural i = 0; i < this->m_Size; ++i )
    this->m_Parameters[ i ] = d( g );
}

// -------------------------------------------------------------------------
template< class _S >
_S& ivqML::Model::Base< _S >::
operator[]( const TNatural& i )
{
  static _S zero = 0;
  if( i < this->m_Size )
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
  if( i < this->m_Size )
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
  return( this->m_Size );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Base< _S >::
set_number_of_parameters( const TNatural& p )
{
  if( this->m_Size != p )
  {
    if( this->m_Parameters != nullptr )
      std::free( this->m_Parameters );

    this->m_Size = p;
    if( this->m_Size > 0 )
      this->m_Parameters =
        reinterpret_cast< TScalar* >(
          std::calloc( this->m_Size, sizeof( TScalar ) )
          );
    else
      this->m_Parameters = nullptr;
  } // end if
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Base< _S >::
_map( TMap& map, const TNatural& r, const TNatural& c, const TNatural& o )
{
  if( this->m_Parameters != nullptr )
    new( &map ) TMap( this->m_Parameters + o, r, c );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Base< _S >::
_from_stream( std::istream& i )
{
  TNatural p;
  i >> p;
  this->set_number_of_parameters( p );
  for( TNatural j = 0; j < this->m_Size; ++j )
    i >> this->m_Parameters[ j ];
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Base< _S >::
_to_stream( std::ostream& o ) const
{
  o << this->m_Size;
  for( TNatural i = 0; i < this->m_Size; ++i )
    o << " " << this->m_Parameters[ i ];
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Base< float >;
template class ivqML_EXPORT ivqML::Model::Base< double >;
template class ivqML_EXPORT ivqML::Model::Base< long double >;

// eof - $RCSfile$
