// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__ITK__MixtureOfGaussiansImageFilter__hxx__
#define __ivqML__ITK__MixtureOfGaussiansImageFilter__hxx__

#include <ivq/ITK/EigenUtils.h>
#include <ivqML/Common/MixtureOfGaussians.h>

// -------------------------------------------------------------------------
template< class _TInImage, class _TLabel, class _TReal >
void ivqML::ITK::MixtureOfGaussiansImageFilter< _TInImage, _TLabel, _TReal >::
SetDebug( TDebug d )
{
  this->m_Debug = d;
}

// -------------------------------------------------------------------------
template< class _TInImage, class _TLabel, class _TReal >
ivqML::ITK::MixtureOfGaussiansImageFilter< _TInImage, _TLabel, _TReal >::
MixtureOfGaussiansImageFilter( )
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
void ivqML::ITK::MixtureOfGaussiansImageFilter< _TInImage, _TLabel, _TReal >::
GenerateData( )
{
  this->AllocateOutputs( );
  auto I
    =
    ivq::ITK::ImageToMatrix( this->GetInput( ) ).
    template cast< TReal >( ).transpose( );
  auto O = ivq::ITK::ImageToMatrix( this->GetOutput( ) ).transpose( );

  this->m_Means = TMatrix::Zero( this->m_NumberOfMeans, I.cols( ) );

  using namespace ivqML::Common::MixtureOfGaussians;
  Init( this->m_Means, I, this->m_InitMethod );
  Fit( this->m_Means, this->m_Covariances, I, this->m_Debug );
  Label( O, I, this->m_Means, this->m_Covariances );
}

#endif // __ivqML__ITK__MixtureOfGaussiansImageFilter__hxx__

// eof - $RCSfile$
