// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>

/* TODO
   #include <ivq/ITK/ColorImageToChannelsImageFilter.h>
   #include <ivqML/ITK/PCAImageFilter.h>
*/

#include <itkCoocurrenceTextureFeaturesImageFilter.h>
#include <itkRunLengthTextureFeaturesImageFilter.h>
#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkVectorContainer.h>
#include <itkVectorImage.h>

#include <ivq/ITK/EigenUtils.h>
#include <ivq/ITK/ImageFileReader.h>

#include <ivqML/Common/PCA.h>

const unsigned int Dim = 2;
using TReal = double;
using TImage = itk::Image< TReal, Dim >;
using TVectorImage = itk::VectorImage< TReal, Dim >;

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

  // Some parameters
  unsigned int bins = 100;

  itk::Size< Dim > radius;
  radius.Fill( 1 );

  // Read image
  auto reader = ivq::ITK::ImageFileReader< TImage >::New( );
  reader->SetFileName( input_image );
  reader->Update( );

  // Get input range
  auto R = ivq::ITK::ImageToMatrix( reader->GetOutput( ) );
  auto minV = R.minCoeff( );
  auto maxV = R.maxCoeff( );

  // Coocurrence-based features
  auto ctf = itk::Statistics::CoocurrenceTextureFeaturesImageFilter< TImage, TVectorImage >::New( );
  ctf->SetInput( reader->GetOutput( ) );
  ctf->SetNumberOfBinsPerAxis( bins );
  ctf->SetHistogramMinimum( minV );
  ctf->SetHistogramMaximum( maxV );
  ctf->SetNeighborhoodRadius( radius );
  ctf->Update( );

  // Run length-based features
  auto rl = itk::Statistics::RunLengthTextureFeaturesImageFilter< TImage, TVectorImage >::New( );
  rl->SetInput( reader->GetOutput( ) );
  rl->SetNumberOfBinsPerAxis( bins );
  rl->SetHistogramValueMinimum( minV );
  rl->SetHistogramValueMaximum( maxV );
  rl->SetNeighborhoodRadius( radius );
  rl->Update( );

  // Build eigen objects
  auto Ct = ivq::ITK::ImageToMatrix( ctf->GetOutput( ) );
  auto Rt = ivq::ITK::ImageToMatrix( rl->GetOutput( ) );

  std::cout << bins << std::endl;
  std::cout << minV << " " << maxV << std::endl;
  std::cout << Ct.rowwise( ).minCoeff( ).transpose( ) << std::endl;
  std::cout << Ct.rowwise( ).maxCoeff( ).transpose( ) << std::endl;
  std::cout << Rt.rowwise( ).minCoeff( ).transpose( ) << std::endl;
  std::cout << Rt.rowwise( ).maxCoeff( ).transpose( ) << std::endl;
  std::cout << "---------------------------------------------" << std::endl;

  using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
  TMatrix X( Ct.rows( ) + Rt.rows( ), Ct.cols( ) );
  X << Ct, Rt;

  // Eigen analysis
  auto E = ivqML::Common::EigenAnalysis( X.transpose( ) );
  std::cout << "-----------------------" << std::endl;
  std::cout << std::get< 0 >( E ) << std::endl;
  std::cout << "-----------------------" << std::endl;
  std::cout << std::get< 1 >( E ) << std::endl;
  std::cout << "-----------------------" << std::endl;
  std::cout << ( std::get< 2 >( E ).transpose( ).array( ) / std::get< 2 >( E ).sum( ) ) << std::endl;
  std::cout << "-----------------------" << std::endl;

  // Output image
  auto out = TVectorImage::New( );
  out->SetLargestPossibleRegion( reader->GetOutput( )->GetLargestPossibleRegion( ) );
  out->SetRequestedRegion( reader->GetOutput( )->GetRequestedRegion( ) );
  out->SetBufferedRegion( reader->GetOutput( )->GetBufferedRegion( ) );
  out->SetSpacing( reader->GetOutput( )->GetSpacing( ) );
  out->SetOrigin( reader->GetOutput( )->GetOrigin( ) );
  out->SetDirection( reader->GetOutput( )->GetDirection( ) );
  out->SetNumberOfComponentsPerPixel( X.rows( ) );
  out->Allocate( );
  ivq::ITK::ImageToMatrix( out.GetPointer( ) ) = X;

  auto writer =
    itk::ImageFileWriter< TVectorImage >::New( );
  writer->SetInput( out );
  writer->SetFileName( "texture.mha" );
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

  std::cout << "+++++++++++++++++++++++++++++" << std::endl;
  auto off2 = ctf->GetOffsets( );
  auto off3 = off2->CastToSTLContainer( );
  for( auto l: off3 )
    std::cout << l << " ";
  std::cout << std::endl;
  std::cout << "+++++++++++++++++++++++++++++" << std::endl;

  auto off4 = rl->GetOffsets( );
  auto off5 = off4->CastToSTLContainer( );
  for( auto l: off5 )
    std::cout << l << " ";
  std::cout << std::endl;
  std::cout << "+++++++++++++++++++++++++++++" << std::endl;


  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
