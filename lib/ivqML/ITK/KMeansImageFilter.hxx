// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__ITK__KMeansImageFilter__hxx__
#define __ivqML__ITK__KMeansImageFilter__hxx__

#include <ivq/ITK/EigenUtils.h>
#include <ivqML/Common/KMeans.h>

// -------------------------------------------------------------------------
template< class _TInImage, class _TLabel, class _TReal >
void ivqML::ITK::KMeansImageFilter< _TInImage, _TLabel, _TReal >::
SetDebug( TDebug d )
{
  this->m_Debug = d;
}

// -------------------------------------------------------------------------
template< class _TInImage, class _TLabel, class _TReal >
ivqML::ITK::KMeansImageFilter< _TInImage, _TLabel, _TReal >::
KMeansImageFilter( )
  : Superclass( )
{
  this->m_Debug
    =
    []( const TReal&, const unsigned long long& )
    ->
    bool
    {
      return( false );
    };
}

// -------------------------------------------------------------------------
template< class _TInImage, class _TLabel, class _TReal >
void ivqML::ITK::KMeansImageFilter< _TInImage, _TLabel, _TReal >::
GenerateData( )
{
  this->AllocateOutputs( );
  auto I
    =
    ivq::ITK::ImageToMatrix( this->GetInput( ) ).
    template cast< TReal >( ).transpose( );
  auto O = ivq::ITK::ImageToMatrix( this->GetOutput( ) ).transpose( );

  this->m_Means = TMatrix::Zero( this->m_NumberOfMeans, I.cols( ) );
  ivqML::Common::KMeans::Init( this->m_Means, I, this->m_InitMethod );
  ivqML::Common::KMeans::Fit( this->m_Means, I, this->m_Debug );
  ivqML::Common::KMeans::Label( O, I, this->m_Means );
}

#endif // __ivqML__ITK__KMeansImageFilter__hxx__

// eof - $RCSfile$
