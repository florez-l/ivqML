// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>

#include <itkImageFileWriter.h>
#include <itkVectorImage.h>
#include <ivq/ITK/EigenUtils.h>
#include <ivq/ITK/ImageFileReader.h>
#include <ivqML/Common/KMeans.h>

const unsigned int Dim = 2;
using TReal = float;
using TImage = itk::VectorImage< TReal, Dim >;

int main( int argc, char** argv )
{
  if( argc < 2 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ]  << " input_image"
      << std::endl;
    return( EXIT_FAILURE );
  } // end if
  std::string input_image = argv[ 1 ];

  auto reader = ivq::ITK::ImageFileReader< TImage >::New( );
  reader->SetFileName( input_image );
  reader->Update( );

  auto I = ivq::ITK::ImageToMatrix( reader->GetOutput( ) ).transpose( );
  auto out = TImage::New( );
  out->SetLargestPossibleRegion( reader->GetOutput( )->GetLargestPossibleRegion( ) );
  out->SetRequestedRegion( reader->GetOutput( )->GetRequestedRegion( ) );
  out->SetBufferedRegion( reader->GetOutput( )->GetBufferedRegion( ) );
  out->SetSpacing( reader->GetOutput( )->GetSpacing( ) );
  out->SetOrigin( reader->GetOutput( )->GetOrigin( ) );
  out->SetDirection( reader->GetOutput( )->GetDirection( ) );
  out->SetNumberOfComponentsPerPixel( 1 );
  out->Allocate( );
  auto L = ivq::ITK::ImageToMatrix( out.GetPointer( ) ).transpose( );

  unsigned long long K = 4;
  ivqML::Common::KMeans< TReal > model;
  model.init_random( I, K );
  // model.init_XX( I, K );
  // model.init_Forgy( I, K );
  model.set_debug(
    []( const TReal& mse ) -> bool
    {
      std::cout << "MSE = " << mse << std::endl;
      return( false );
    }
    );
  model.fit( I );
  model.label( L, I );

  auto writer = itk::ImageFileWriter< TImage >::New( );
  writer->SetInput( out );
  writer->SetFileName( "labels.mha" );
  writer->Update( );

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
