// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <sstream>

#include <itkImageFileWriter.h>
#include <itkVectorImage.h>

#include <ivq/ITK/EigenUtils.h>
#include <ivq/ITK/ImageFileReader.h>

#include <ivqML/ITK/PCAImageFilter.h>

const unsigned int Dim = 2;
using TReal = double;
using TImage = itk::VectorImage< TReal, Dim >;

int main( int argc, char** argv )
{
  if( argc < 3 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ]
      << " input_image output_image [kept_information]"
      << std::endl;
    return( EXIT_FAILURE );
  } // end if
  std::string input_image = argv[ 1 ];
  std::string output_image = argv[ 2 ];

  // Some optional parameters
  TReal kept_information = 1;
  if( argc > 3 ) std::istringstream( argv[ 3 ] ) >> kept_information;

  // Read image
  auto reader = ivq::ITK::ImageFileReader< TImage >::New( );
  reader->SetFileName( input_image );

  // PCA
  auto pca = ivqML::ITK::PCAImageFilter< TImage, TReal >::New( );
  pca->SetInput( reader->GetOutput( ) );
  pca->SetKeptInformation( kept_information );

  // Write result
  auto writer =
    itk::ImageFileWriter< decltype( pca )::ObjectType::TOutImage >::New( );
  writer->SetInput( pca->GetOutput( ) );
  writer->SetFileName( output_image );
  try
  {
    writer->Update( );
  }
  catch( std::exception& err )
  {
    std::cerr << "Error caught: " << err.what( ) << std::endl;
    return( EXIT_FAILURE );
  } // end try

  std::cout << "******** RESULTS ********" << std::endl;
  unsigned int d = pca->GetOutput( )->GetNumberOfComponentsPerPixel( );
  std::cout
    << "Kept dimensions : "
    << d
    << std::endl;
  std::cout
    << "Kept information: "
    << ( pca->GetValues( )( 0, d - 1 ) * 100 ) << "%"
    << std::endl;
  std::cout << "*************************" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
