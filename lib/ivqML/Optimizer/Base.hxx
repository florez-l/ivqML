// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Base__hxx__
#define __ivqML__Optimizer__Base__hxx__

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Optimizer::Base< _TModel >::
Base( TModel& m )
  : m_Model( &m )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Optimizer::Base< _TModel >::
~Base( )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
typename ivqML::Optimizer::Base< _TModel >::
TModel* ivqML::Optimizer::Base< _TModel >::
model( ) const
{
  return( this->m_Model );
}

// -------------------------------------------------------------------------
template< class _TModel >
const typename ivqML::Optimizer::Base< _TModel >::
TReal& ivqML::Optimizer::Base< _TModel >::
lambda1( ) const
{
  return( this->m_Lambda1 );
}

// -------------------------------------------------------------------------
template< class _TModel >
const typename ivqML::Optimizer::Base< _TModel >::
TReal& ivqML::Optimizer::Base< _TModel >::
lambda2( ) const
{
  return( this->m_Lambda2 );
}

// -------------------------------------------------------------------------
template< class _TModel >
const typename ivqML::Optimizer::Base< _TModel >::
TReal& ivqML::Optimizer::Base< _TModel >::
epsilon( ) const
{
  return( this->m_Epsilion );
}

// -------------------------------------------------------------------------
template< class _TModel >
void  ivqML::Optimizer::Base< _TModel >::
setLambda1( const TReal& l )
{
  this->m_Lambda1 = l;
}

// -------------------------------------------------------------------------
template< class _TModel >
void ivqML::Optimizer::Base< _TModel >::
setLambda2( const TReal& l )
{
  this->m_Lambda2 = l;
}

// -------------------------------------------------------------------------
template< class _TModel >
void ivqML::Optimizer::Base< _TModel >::
setEpsilon( const TReal& e )
{
  this->m_Epsilon = e;
}

// -------------------------------------------------------------------------
template< class _TModel >
void ivqML::Optimizer::Base< _TModel >::
setDebug( TDebug d )
{
  this->m_Debug = d;
}

#endif // __ivqML__Optimizer__Base__hxx__

// eof - $RCSfile$
