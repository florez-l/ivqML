// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Base.h>
#include <cstdlib>
#include <random>

// -------------------------------------------------------------------------
template< class _TScalar >
ivqML::Model::Base< _TScalar >::
Base( const TNatural& n )
{
  this->set_number_of_parameters( n );
}

// -------------------------------------------------------------------------
template< class _TScalar >
ivqML::Model::Base< _TScalar >::
~Base( )
{
  if( this->m_Parameters != nullptr )
  {
    std::free( this->m_Parameters );
    this->m_Parameters = nullptr;
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScalar >
void ivqML::Model::Base< _TScalar >::
random_fill( )
{
  std::random_device r;
  std::mt19937 g( r( ) );
  std::uniform_real_distribution< TScalar > d( 0, 1 );
  for( TNatural i = 0; i < this->m_Size; ++i )
    this->m_Parameters[ i ] = d( g );
}

// -------------------------------------------------------------------------
template< class _TScalar >
_TScalar& ivqML::Model::Base< _TScalar >::
operator[]( const TNatural& i )
{
  static TScalar zero = TScalar( 0 );
  if( i < this->m_Size )
    return( this->m_Parameters[ i ] );
  else
  {
    zero = TScalar( 0 );
    return( zero );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TScalar >
const _TScalar& ivqML::Model::Base< _TScalar >::
operator[]( const TNatural& i ) const
{
  static const TScalar zero = TScalar( 0 );
  if( i < this->m_Size )
    return( this->m_Parameters[ i ] );
  else
    return( zero );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::Base< _TScalar >::
TNatural ivqML::Model::Base< _TScalar >::
buffer_size( ) const
{
  return( this->number_of_parameters( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::Base< _TScalar >::
TNatural ivqML::Model::Base< _TScalar >::
number_of_parameters( ) const
{
  return( this->m_Size );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void ivqML::Model::Base< _TScalar >::
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
template< class _TScalar >
Eigen::Map< typename ivqML::Model::Base< _TScalar >::TMatrix >
ivqML::Model::Base< _TScalar >::
_matrix( const TNatural& r, const TNatural& c, const TNatural& o )
{
  if( this->m_Parameters != nullptr )
    if( ( r * c ) <= ( this->m_Size - o ) )
      return( Eigen::Map< TMatrix >( this->m_Parameters + o, r, c ) );
  return( Eigen::Map< TMatrix >( nullptr, 0, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
Eigen::Map< const typename ivqML::Model::Base< _TScalar >::TMatrix >
ivqML::Model::Base< _TScalar >::
_matrix( const TNatural& r, const TNatural& c, const TNatural& o ) const
{
  if( this->m_Parameters != nullptr )
    if( ( r * c ) <= ( this->m_Size - o ) )
      return( Eigen::Map< const TMatrix >( this->m_Parameters + o, r, c ) );
  return( Eigen::Map< const TMatrix >( nullptr, 0, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
Eigen::Map< typename ivqML::Model::Base< _TScalar >::TColumn >
ivqML::Model::Base< _TScalar >::
_column( const TNatural& r, const TNatural& o )
{
  if( this->m_Parameters != nullptr )
    if( r <= ( this->m_Size - o ) )
      return( Eigen::Map< TColumn >( this->m_Parameters + o, r, 1 ) );
  return( Eigen::Map< TColumn >( nullptr, 0, 1 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
Eigen::Map< const typename ivqML::Model::Base< _TScalar >::TColumn >
ivqML::Model::Base< _TScalar >::
_column( const TNatural& r, const TNatural& o ) const
{
  if( this->m_Parameters != nullptr )
    if( r <= ( this->m_Size - o ) )
      return( Eigen::Map< const TColumn >( this->m_Parameters + o, r, 1 ) );
  return( Eigen::Map< const TColumn >( nullptr, 0, 1 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
Eigen::Map< typename ivqML::Model::Base< _TScalar >::TRow >
ivqML::Model::Base< _TScalar >::
_row( const TNatural& c, const TNatural& o )
{
  if( this->m_Parameters != nullptr )
    if( c <= ( this->m_Size - o ) )
      return( Eigen::Map< TRow >( this->m_Parameters + o, 1, c ) );
  return( Eigen::Map< TRow >( nullptr, 1, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
Eigen::Map< const typename ivqML::Model::Base< _TScalar >::TRow >
ivqML::Model::Base< _TScalar >::
_row( const TNatural& c, const TNatural& o ) const
{
  if( this->m_Parameters != nullptr )
    if( c <= ( this->m_Size - o ) )
      return( Eigen::Map< const TRow >( this->m_Parameters + o, 1, c ) );
  return( Eigen::Map< const TRow >( nullptr, 1, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void ivqML::Model::Base< _TScalar >::
_from_stream( std::istream& i )
{
  TNatural p;
  i >> p;
  this->set_number_of_parameters( p );
  for( TNatural j = 0; j < this->m_Size; ++j )
    i >> this->m_Parameters[ j ];
}

// -------------------------------------------------------------------------
template< class _TScalar >
void ivqML::Model::Base< _TScalar >::
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
