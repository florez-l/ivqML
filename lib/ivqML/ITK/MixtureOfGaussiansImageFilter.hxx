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
  this->m_Debug = []( const TReal& mse ) -> bool { return( false ); };
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

  ivqML::Common::MixtureOfGaussians< TReal > model;
  model.init_random( I, this->m_NumberOfMeans );
  model.set_debug( this->m_Debug );
  model.fit( I );
  model.label( O, I );
}

#endif // __ivqML__ITK__MixtureOfGaussiansImageFilter__hxx__

// eof - $RCSfile$
