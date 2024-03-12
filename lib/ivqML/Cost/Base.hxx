// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__Base__hxx__
#define __ivqML__Cost__Base__hxx__

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Cost::Base< _TModel >::
Base( _TModel& m )
  : m_M( &m )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Cost::Base< _TModel >::
~Base( )
{
  if( this->m_B != nullptr )
    delete this->m_B;
}

// -------------------------------------------------------------------------
template< class _TModel >
template< class _TInputX, class _TInputY >
void ivqML::Cost::Base< _TModel >::
set_data(
  const Eigen::EigenBase< _TInputX >& iX,
  const Eigen::EigenBase< _TInputY >& iY
  )
{
  this->m_X = iX.derived( ).template cast< TScalar >( );
  this->m_Y = iY.derived( ).template cast< TScalar >( );
}

#endif // __ivqML__Cost__Base__hxx__

// eof - $RCSfile$
