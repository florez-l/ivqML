// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>

#include <itkImageFileWriter.h>
#include <itkVectorImage.h>
#include <ivq/ITK/ColorImageToChannelsImageFilter.h>
#include <ivq/ITK/ImageFileReader.h>
#include <ivqML/ITK/PCAImageFilter.h>


const unsigned int Dim = 2;
using TReal = float;
using TPixel = unsigned char;
using TImage = itk::VectorImage< TPixel, Dim >;

int main( int argc, char** argv )
{
  if( argc < 3 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ]  << " input_image output_image"
      << std::endl;
    return( EXIT_FAILURE );
  } // end if
  std::string input_image = argv[ 1 ];
  std::string output_image = argv[ 2 ];

  auto reader = ivq::ITK::ImageFileReader< TImage >::New( );
  reader->SetFileName( input_image );
  reader->Update( );

  unsigned int c = reader->GetOutput( )->GetNumberOfComponentsPerPixel( );
  if( c != 3 && c != 4 )
  {
    std::cerr
      << "Input image does not have color information "
      << "(number_of_channels=" << c << ")" << std::endl;
    return( EXIT_FAILURE );
  } // end if

  auto filter =
    ivq::ITK::ColorImageToChannelsImageFilter< TImage, TReal >::New( );
  filter->SetInput( reader->GetOutput( ) );
  if( argc > 3 )
  {
    for( int i = 3; i < argc; ++i )
      filter->UseChannel( argv[ i ] );
  }
  else
    filter->UseAllChannels( );

  auto pca =
    ivqML::ITK::PCAImageFilter< decltype( filter )::ObjectType::TOutImage, TReal >::New( );
  pca->SetInput( filter->GetOutput( ) );

  auto writer =
    itk::ImageFileWriter< decltype( pca )::ObjectType::TOutImage >::New( );
  writer->SetInput( pca->GetOutput( ) );
  writer->SetFileName( output_image );
  writer->UseCompressionOn( );
  try
  {
    writer->Update( );
  }
  catch( std::exception& err )
  {
    std::cerr << "Error caught: " << err.what( ) << std::endl;
    return( EXIT_FAILURE );
  } // end try
  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
