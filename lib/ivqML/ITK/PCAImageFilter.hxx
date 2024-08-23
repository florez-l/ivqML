// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__ITK__PCAImageFilter__hxx__
#define __ivqML__ITK__PCAImageFilter__hxx__

#include <iterator>
#include <numeric>
#include <ivq/ITK/EigenUtils.h>
#include <ivqML/Common/PCA.h>

// -------------------------------------------------------------------------
template< class _TInImage, class _TReal >
void ivqML::ITK::PCAImageFilter< _TInImage, _TReal >::
SetNumberOfKeptDimensions( const unsigned int& i )
{
  if( i > 0 )
  {
    if( i == 1 )
    {
      if( this->m_KeptInformation != -1 )
      {
        this->m_KeptInformation = -1;
        this->Modified( );
      } // end if
    }
    else
    {
      if( ( unsigned int )( std::fabs( this->m_KeptInformation ) ) != i )
      {
        this->m_KeptInformation = TReal( i );
        this->Modified( );
      } // end if
    } // end if
  } // end if
}

// -------------------------------------------------------------------------
template< class _TInImage, class _TReal >
ivqML::ITK::PCAImageFilter< _TInImage, _TReal >::
PCAImageFilter( )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _TInImage, class _TReal >
void ivqML::ITK::PCAImageFilter< _TInImage, _TReal >::
GenerateOutputInformation( )
{
  TInImage* in = const_cast< TInImage* >( this->GetInput( ) );
  in->Update( );

  auto X = ivq::ITK::ImageToMatrix( in ).transpose( );
  auto E = ivqML::Common::EigenAnalysis< decltype( X ), TReal >( X );

  this->m_Mean     = std::get< 0 >( E ).transpose( );
  this->m_Rotation = std::get< 1 >( E ).transpose( );
  this->m_Values   =
    ( std::get< 2 >( E ).transpose( ).array( ) / std::get< 2 >( E ).sum( ) );

  std::partial_sum(
    this->m_Values.data( ), this->m_Values.data( ) + this->m_Values.size( ),
    this->m_Values.data( )
    );

  unsigned int c = 0;
  if(
    this->m_KeptInformation >= TReal( 0 )
    &&
    this->m_KeptInformation <= TReal( 1 )
    )
    c = 1 + std::distance(
      this->m_Values.data( ),
      std::find_if_not(
        this->m_Values.data( ),
        this->m_Values.data( ) + this->m_Values.size( ),
        [&]( const TReal& v )
        {
          return( v < this->m_KeptInformation );
        }
        )
      );
  else
    c = ( unsigned int )( std::fabs( this->m_KeptInformation ) );
  c = ( c == 0 )? 1: c;
  c =
    ( in->GetNumberOfComponentsPerPixel( ) < c )
    ?
    in->GetNumberOfComponentsPerPixel( )
    :
    c;

  TOutImage* out = this->GetOutput( );
  out->SetLargestPossibleRegion( in->GetLargestPossibleRegion( ) );
  out->SetRequestedRegion( in->GetRequestedRegion( ) );
  out->SetSpacing( in->GetSpacing( ) );
  out->SetOrigin( in->GetOrigin( ) );
  out->SetDirection( in->GetDirection( ) );
  out->SetNumberOfComponentsPerPixel( c );
}

// -------------------------------------------------------------------------
template< class _TInImage, class _TReal >
void ivqML::ITK::PCAImageFilter< _TInImage, _TReal >::
GenerateData( )
{
  this->AllocateOutputs( );
  const TInImage* in = this->GetInput( );
  TOutImage* out = this->GetOutput( );

  auto I = ivq::ITK::ImageToMatrix( in ).template cast< TReal >( );
  auto O = ivq::ITK::ImageToMatrix( out );

  O = ( this->m_Rotation * ( I.colwise( ) - this->m_Mean.col( 0 ) ) )
    .block( 0, 0, O.rows( ), O.cols( ) );
}

#endif // __ivqML__ITK__PCAImageFilter__hxx__

// eof - $RCSfile$
