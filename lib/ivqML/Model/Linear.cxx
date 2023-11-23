// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Linear.h>

// -------------------------------------------------------------------------
template< class _S >
ivqML::Model::Linear< _S >::
Linear( const TNatural& n )
  : Superclass( TNatural( 0 ) )
{
  this->set_number_of_parameters( n + 1 );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::Linear< _S >::
TNatural ivqML::Model::Linear< _S >::
number_of_inputs( ) const
{
  return( this->m_P - 1 );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Linear< _S >::
set_number_of_inputs( const TNatural& i )
{
  this->set_number_of_parameters( i + 1 );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::Linear< _S >::
TNatural ivqML::Model::Linear< _S >::
number_of_outputs( ) const
{
  return( 1 );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Linear< _S >::
_synch( )
{
  if( this->m_P > 0 )
    new( &this->m_M ) TMap( this->m_T.get( ) + 1, this->m_P - 1, 1 );
  else
    new( &this->m_M ) TMap( nullptr, 0, 0 );
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Linear< float >;
template class ivqML_EXPORT ivqML::Model::Linear< double >;
template class ivqML_EXPORT ivqML::Model::Linear< long double >;

// eof - $RCSfile$
