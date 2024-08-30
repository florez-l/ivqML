// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__ITK__KMeansImageFilter__hxx__
#define __ivqML__ITK__KMeansImageFilter__hxx__

#include <ivq/ITK/EigenUtils.h>
#include <ivqML/Common/KMeans.h>

// -------------------------------------------------------------------------
template< class _TInImage, class _TLabel, class _TReal >
ivqML::ITK::KMeansImageFilter< _TInImage, _TLabel, _TReal >::
KMeansImageFilter( )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _TInImage, class _TLabel, class _TReal >
void ivqML::ITK::KMeansImageFilter< _TInImage, _TLabel, _TReal >::
GenerateData( )
{
  this->AllocateOutputs( );
  /* TODO
     unsigned long long m_NumberOfMeans { 2 };
     TMatrix m_Means;
  */
}

#endif // __ivqML__ITK__KMeansImageFilter__hxx__

// eof - $RCSfile$
