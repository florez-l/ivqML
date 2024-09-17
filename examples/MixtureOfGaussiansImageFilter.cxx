// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <sstream>

#include <itkImageFileWriter.h>
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
      << "Usage: " << argv[ 0 ]
      << " input_image output_image [K=2] [max_iter=200]"
      << std::endl;
    return( EXIT_FAILURE );
  } // end if
  std::string input_image = argv[ 1 ];
  std::string output_image = argv[ 2 ];
  unsigned long long K = 2;
  unsigned long long max_iter = 200;
  if( argc > 3 ) std::istringstream( argv[ 3 ] ) >> K;
  if( argc > 4 ) std::istringstream( argv[ 4 ] ) >> max_iter;

  auto reader = ivq::ITK::ImageFileReader< TImage >::New( );
  reader->SetFileName( input_image );

  auto mog = ivqML::ITK::MixtureOfGaussiansImageFilter< TImage >::New( );
  mog->SetInput( reader->GetOutput( ) );
  mog->SetNumberOfMeans( K );
  mog->SetInitMethod( "++" );
  mog->SetDebug(
    [&]( const TReal& mse, const unsigned long long& iter ) -> bool
    {
      std::cout << iter << " " << mse << std::endl;
      return( iter == max_iter );
    }
    );

  auto writer
    =
    itk::ImageFileWriter< decltype( mog )::ObjectType::TOutImage >::New( );
  writer->SetInput( mog->GetOutput( ) );
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
