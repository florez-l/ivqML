// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Adam__hxx__
#define __ivqML__Optimizer__Adam__hxx__

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Optimizer::Adam< _TModel >::
Adam( )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Optimizer::Adam< _TModel >::
~Adam( )
{
}
// -------------------------------------------------------------------------
template< class _TModel >
void ivqML::Optimizer::Adam< _TModel >::
_fit( TModel& model, TBatches& batches, TShuffler shuffler )
{
}

#endif // __ivqML__Optimizer__Adam__hxx__

// eof - $RCSfile$
