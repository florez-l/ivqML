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
  if( this->m_T != nullptr )
    delete this->m_T;
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Base< _S >::
random_fill( )
{
  std::random_device r;
  std::mt19937 g( r( ) );
  std::uniform_real_distribution< _S > d( 0, 1 );
  for( TNatural i = 0; i < this->m_P; ++i )
    *( this->m_T + i ) = d( g );
}

// -------------------------------------------------------------------------
template< class _S >
_S& ivqML::Model::Base< _S >::
operator[]( const TNatural& i )
{
  static _S zero = 0;
  if( i < this->m_P )
    return( *( this->m_T + i ) );
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
  if( i < this->m_P )
    return( *( this->m_T + i ) );
  else
    return( zero );
}

// -------------------------------------------------------------------------
template< class _S >
const typename ivqML::Model::Base< _S >::
TNatural& ivqML::Model::Base< _S >::
number_of_parameters( ) const
{
  return( this->m_P );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Base< _S >::
set_number_of_parameters( const TNatural& p )
{
  if( this->m_P != p )
  {
    if( this->m_T != nullptr )
      delete this->m_T;
    this->m_T = ( p > 0 )? new _S[ p ]: nullptr;
    this->m_P = p;
  } // end if
  if( p > 0 )
    std::memset( this->m_T, 0, p * sizeof( _S ) );
}

// -------------------------------------------------------------------------
template< class _S >
_S* ivqML::Model::Base< _S >::
begin( )
{
  return( this->m_T );
}

// -------------------------------------------------------------------------
template< class _S >
const _S* ivqML::Model::Base< _S >::
begin( ) const
{
  return( this->m_T );
}

// -------------------------------------------------------------------------
template< class _S >
_S* ivqML::Model::Base< _S >::
end( )
{
  return( this->m_T + this->m_P );
}

// -------------------------------------------------------------------------
template< class _S >
const _S* ivqML::Model::Base< _S >::
end( ) const
{
  return( this->m_T + this->m_P );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Base< _S >::
_to_stream( std::ostream& o ) const
{
  o << this->m_P;
  for( TNatural i = 0; i < this->m_P; ++i )
    o << " " << *( this->m_T + i );
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Base< float >;
template class ivqML_EXPORT ivqML::Model::Base< double >;
template class ivqML_EXPORT ivqML::Model::Base< long double >;

// eof - $RCSfile$
