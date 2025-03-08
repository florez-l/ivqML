// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Base.h>
#include <cstring>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
ivqML::Model::Base< _TReal, _TNatural >::
Base( const TNatural& n )
{
  this->_resize( n );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
ivqML::Model::Base< _TReal, _TNatural >::
~Base( )
{
  this->_resize( 0 );
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
init( )
{
  TColMap( this->m_P, this->m_S, 0 ) *= TReal( 0 );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::Base< _TReal, _TNatural >::
_resize( const TNatural& n )
{
  if( this->m_P != nullptr )
    std::free( this->m_P );
  this->m_S = n;
  this->m_P = nullptr;
  if( n > 0 )
    this->m_P
      =
      reinterpret_cast< TReal* >( std::calloc( n, sizeof( TReal ) ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::Base< _TReal, _TNatural >::
_to_stream( std::ostream& o ) const
{
  o << this->m_S;
  for( TNatural i = 0; i < this->m_S; ++i )
    o << " " << this->m_P[ i ];
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

// eof - Base.cxx
