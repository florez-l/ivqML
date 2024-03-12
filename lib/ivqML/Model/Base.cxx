// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Base.h>
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
void ivqML::Model::Base< _TScalar >::
random_fill( )
{
  std::random_device r;
  std::mt19937 g( r( ) );
  std::uniform_real_distribution< TScalar > d( -1, 1 );
  for( TNatural i = 0; i < this->m_P.size( ); ++i )
    this->m_P( i ) = d( g );
}

// -------------------------------------------------------------------------
template< class _TScalar >
_TScalar& ivqML::Model::Base< _TScalar >::
operator[]( const TNatural& i )
{
  static TScalar zero = TScalar( 0 );
  if( i < this->m_P.size( ) )
    return( this->m_P( i ) );
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
  if( i < this->m_P.size( ) )
    return( this->m_P( i ) );
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
  return( this->m_P.size( ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void ivqML::Model::Base< _TScalar >::
set_number_of_parameters( const TNatural& p )
{
  this->m_P = TCol::Zero( p );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::Base< _TScalar >::
TMatMap ivqML::Model::Base< _TScalar >::
matrix( const TNatural& r, const TNatural& c, const TNatural& o )
{
  if( ( r * c ) <= ( this->m_P.size( ) - o ) )
    return( TMatMap( this->m_P.data( ) + o, r, c ) );
  return( TMatMap( nullptr, 0, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::Base< _TScalar >::
TMatCMap ivqML::Model::Base< _TScalar >::
matrix( const TNatural& r, const TNatural& c, const TNatural& o ) const
{
  if( ( r * c ) <= ( this->m_P.size( ) - o ) )
    return( TMatCMap( this->m_P.data( ) + o, r, c ) );
  return( TMatCMap( nullptr, 0, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::Base< _TScalar >::
TColMap ivqML::Model::Base< _TScalar >::
column( const TNatural& r, const TNatural& o )
{
  if( r <= ( this->m_P.size( ) - o ) )
    return( TColMap( this->m_P.data( ) + o, r, 1 ) );
  return( TColMap( nullptr, 0, 1 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::Base< _TScalar >::
TColCMap ivqML::Model::Base< _TScalar >::
column( const TNatural& r, const TNatural& o ) const
{
  if( r <= ( this->m_P.size( ) - o ) )
    return( TColCMap( this->m_P.data( ) + o, r, 1 ) );
  return( TColCMap( nullptr, 0, 1 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::Base< _TScalar >::
TRowMap ivqML::Model::Base< _TScalar >::
row( const TNatural& c, const TNatural& o )
{
  if( c <= ( this->m_P.size( ) - o ) )
    return( TRowMap( this->m_P.data( ) + o, 1, c ) );
  return( TRowMap( nullptr, 1, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename ivqML::Model::Base< _TScalar >::
TRowCMap ivqML::Model::Base< _TScalar >::
row( const TNatural& c, const TNatural& o ) const
{
  if( c <= ( this->m_P.size( ) - o ) )
    return( TRowCMap( this->m_P.data( ) + o, 1, c ) );
  return( TRowCMap( nullptr, 1, 0 ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
bool ivqML::Model::Base< _TScalar >::
has_backpropagation( ) const
{
  return( false );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void ivqML::Model::Base< _TScalar >::
_from_stream( std::istream& i )
{
  TNatural p;
  i >> p;
  this->set_number_of_parameters( p );
  for( TNatural j = 0; j < this->m_P.size( ); ++j )
    i >> this->m_P( j );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void ivqML::Model::Base< _TScalar >::
_to_stream( std::ostream& o ) const
{
  o << this->m_P.size( );
  for( TNatural i = 0; i < this->m_P.size( ); ++i )
    o << " " << this->m_P( i );
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Base< float >;
template class ivqML_EXPORT ivqML::Model::Base< double >;
template class ivqML_EXPORT ivqML::Model::Base< long double >;

// eof - $RCSfile$
