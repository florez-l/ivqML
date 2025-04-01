// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Base.h>

#include <algorithm>
#include <cstring>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
ivqML::Model::Base< _TReal, _TNatural >::
Base( const TNatural& n )
{
  this->set_size( n );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
ivqML::Model::Base< _TReal, _TNatural >::
~Base( )
{
  if( this->m_P != nullptr )
    std::free( this->m_P );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
const typename ivqML::Model::Base< _TReal, _TNatural >::
TNatural& ivqML::Model::Base< _TReal, _TNatural >::
size( ) const
{
  return( this->m_S );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::Base< _TReal, _TNatural >::
set_size( const TNatural& n )
{
  if( this->m_S != n )
  {
    this->m_S = n;
    this->init( );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::Base< _TReal, _TNatural >::
init( std::function< TReal( ) > g )
{
  if( this->m_P != nullptr )
    std::free( this->m_P );
  this->m_P = nullptr;
  if( this->m_S > 0 )
    this->m_P
      =
      reinterpret_cast< TReal* >( std::calloc( this->m_S, sizeof( TReal ) ) );
  if( this->m_P != nullptr )
    std::generate( this->m_P, this->m_P + this->m_S, g );
  else
    this->m_S = 0;
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::Base< _TReal, _TNatural >::
_to_stream( std::ostream& o ) const
{
  o << this->m_S;
  for( TNatural i = 0; i < this->m_S; ++i )
    o << " " << *( this->m_P + i );
}

// -------------------------------------------------------------------------
namespace ivqML
{
  namespace Model
  {
    template class ivqML_EXPORT Base< float, unsigned int >;
    template class ivqML_EXPORT Base< float, unsigned long >;
    template class ivqML_EXPORT Base< float, unsigned long long >;
    template class ivqML_EXPORT Base< double, unsigned int >;
    template class ivqML_EXPORT Base< double, unsigned long >;
    template class ivqML_EXPORT Base< double, unsigned long long >;
    template class ivqML_EXPORT Base< long double, unsigned int >;
    template class ivqML_EXPORT Base< long double, unsigned long >;
    template class ivqML_EXPORT Base< long double, unsigned long long >;
  } // end namespace
} // end namespace

// eof - $RCSfile$
