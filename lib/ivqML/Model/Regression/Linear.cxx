// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Regression/Linear.h>

// -------------------------------------------------------------------------
template< class _S >
ivqML::Model::Regression::Linear< _S >::
Linear( const TNatural& n )
  : Superclass( TNatural( 0 ) )
{
  this->set_number_of_parameters( n + 1 );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Regression::Linear< _S >::
set_number_of_parameters( const TNatural& p )
{
  this->Superclass::set_number_of_parameters( p );
  this->_map( this->m_T, 1, p - 1, 1 );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::Regression::Linear< _S >::
TNatural ivqML::Model::Regression::Linear< _S >::
number_of_inputs( ) const
{
  return( this->number_of_parameters( ) - 1 );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Regression::Linear< _S >::
set_number_of_inputs( const TNatural& i )
{
  this->set_number_of_parameters( i + 1 );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::Regression::Linear< _S >::
TNatural ivqML::Model::Regression::Linear< _S >::
number_of_outputs( ) const
{
  return( 1 );
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Regression::Linear< float >;
template class ivqML_EXPORT ivqML::Model::Regression::Linear< double >;
template class ivqML_EXPORT ivqML::Model::Regression::Linear< long double >;

// eof - $RCSfile$
