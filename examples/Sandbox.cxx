// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <sstream>

#include <itkImageFileWriter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkVectorImage.h>
#include <ivq/ITK/ImageFileReader.h>
#include <ivqML/ITK/MixtureOfGaussiansImageFilter.h>

const unsigned int Dim = 2;
using TReal = float;
using TImage = itk::VectorImage< TReal, Dim >;

int main( int argc, char** argv )
{
  if( argc < 3 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ]  << " input_image output_image [K=2]"
      << std::endl;
    return( EXIT_FAILURE );
  } // end if
  std::string input_image = argv[ 1 ];
  std::string output_image = argv[ 2 ];
  unsigned long long K = 2;
  if( argc > 3 ) std::istringstream( argv[ 3 ] ) >> K;

  auto reader = ivq::ITK::ImageFileReader< TImage >::New( );
  reader->SetFileName( input_image );

  auto mog = ivqML::ITK::MixtureOfGaussiansImageFilter< TImage >::New( );
  mog->SetInput( reader->GetOutput( ) );
  mog->SetNumberOfMeans( K );
  mog->SetDebug(
    []( const TReal& mse ) -> bool
    {
      std::cout << "MSE = " << mse << std::endl;
      return( false );
    }
    );

  using TLabels = decltype( mog )::ObjectType::TOutImage;
  using TLabel = itk::NumericTraits< TLabels::PixelType >::ValueType;

  auto rescaler = itk::RescaleIntensityImageFilter< TLabels, TLabels >::New( );
  rescaler->SetInput( mog->GetOutput( ) );
  rescaler->SetOutputMinimum( std::numeric_limits< TLabel >::min( ) );
  rescaler->SetOutputMaximum( std::numeric_limits< TLabel >::max( ) );

  auto writer = itk::ImageFileWriter< TLabels >::New( );
  writer->SetInput( rescaler->GetOutput( ) );
  writer->SetFileName( output_image );
  try
  {
    writer->Update( );
  }
  catch( const std::exception& err )
  {
    std::cerr << "Error caught: " << err.what( ) << std::endl;
    return( EXIT_FAILURE );
  } // end try

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
