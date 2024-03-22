// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Regression/Linear.h>

// -------------------------------------------------------------------------
template< class _TScl >
ivqML::Model::Regression::Linear< _TScl >::
Linear( const TNat& n )
  : Superclass( TNat( 0 ) )
{
  this->set_number_of_parameters( n + 1 );
}

// -------------------------------------------------------------------------
template< class _TScl >
void ivqML::Model::Regression::Linear< _TScl >::
set_number_of_parameters( const TNat& p )
{
  this->Superclass::set_number_of_parameters( p );
  if( p > 0 )
    new ( &this->m_T ) TRowMap( &( this->operator[]( 1 ) ), 1, p - 1 );
  else
    new ( &this->m_T ) TRowMap( nullptr, 0, 0 );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ivqML::Model::Regression::Linear< _TScl >::
TNat ivqML::Model::Regression::Linear< _TScl >::
number_of_inputs( ) const
{
  return( this->number_of_parameters( ) - 1 );
}

// -------------------------------------------------------------------------
template< class _TScl >
void ivqML::Model::Regression::Linear< _TScl >::
set_number_of_inputs( const TNat& i )
{
  this->set_number_of_parameters( i + 1 );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ivqML::Model::Regression::Linear< _TScl >::
TNat ivqML::Model::Regression::Linear< _TScl >::
number_of_outputs( ) const
{
  return( 1 );
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Regression::Linear< float >;
template class ivqML_EXPORT ivqML::Model::Regression::Linear< double >;
template class ivqML_EXPORT ivqML::Model::Regression::Linear< long double >;

// eof - $RCSfile$
