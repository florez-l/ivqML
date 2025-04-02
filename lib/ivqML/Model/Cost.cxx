// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Cost.h>

// -------------------------------------------------------------------------
template< class _TReal >
ivqML::Model::Cost< _TReal >::
Cost( )
{
}

// -------------------------------------------------------------------------
template< class _TReal >
ivqML::Model::Cost< _TReal >::
~Cost( )
{
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Model::Cost< _TReal >::
set_type_to_MSE( )
{
  this->m_Type = Self::MSE;
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Model::Cost< _TReal >::
set_type_to_MCE( )
{
  this->m_Type = Self::MCE;
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::Model::Cost< _TReal >::
set_type_to_CCE( )
{
  this->m_Type = Self::CCE;
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Cost< float >;
template class ivqML_EXPORT ivqML::Model::Cost< double >;
template class ivqML_EXPORT ivqML::Model::Cost< long double >;

// eof - $RCSfile$
