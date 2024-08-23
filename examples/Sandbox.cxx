// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

#include <itkImageFileWriter.h>
#include <itkVectorImage.h>
#include <ivq/ITK/EigenUtils.h>
#include <ivq/ITK/ImageFileReader.h>

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

  auto I = ivq::ITK::ImageToMatrix( reader->GetOutput( ) );
  auto out = TImage::New( );
  out->SetLargestPossibleRegion( reader->GetOutput( )->GetLargestPossibleRegion( ) );
  out->SetRequestedRegion( reader->GetOutput( )->GetRequestedRegion( ) );
  out->SetBufferedRegion( reader->GetOutput( )->GetBufferedRegion( ) );
  out->SetSpacing( reader->GetOutput( )->GetSpacing( ) );
  out->SetOrigin( reader->GetOutput( )->GetOrigin( ) );
  out->SetDirection( reader->GetOutput( )->GetDirection( ) );
  out->SetNumberOfComponentsPerPixel( 1 );
  out->Allocate( );
  auto L = ivq::ITK::ImageToMatrix( out.GetPointer( ) );

  std::vector< unsigned long long > idx( I.cols( ) );
  std::iota( idx.begin( ), idx.end( ), 0 );
  unsigned long long seed
    =
    std::chrono::system_clock::now( ).time_since_epoch( ).count( );
  std::shuffle( idx.begin( ), idx.end( ), std::default_random_engine( seed ) );

  using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
  TMatrix centers = I( ivq_EIGEN_ALL, { idx[ 0 ], idx[ 1 ], idx[ 2 ] } );

  unsigned long long nLabels = centers.cols( );
  TMatrix D( nLabels, I.cols( ) );

  for( unsigned long long l = 0; l < nLabels; ++l )
  {
    D.row( l ) =
      ( I.colwise( ) - centers.col( l ) ).array( )
      .pow( 2 ).colwise( ).sum( ).sqrt( );
  } // end if

  for( unsigned long long c = 0; c < L.cols( ); ++c )
    D.col( c ).minCoeff( &L( 0, c ) );

  for( unsigned long long l = 0; l < nLabels; ++l )
  {
    auto J = ( L.array( ) == l ).template cast< unsigned long long >( ).eval( );
    std::cout << J.select( I, 0 ).rows( ) << " " << J.select( I, 0 ).cols( ) << std::endl;
    std::cout << "---------------------" << std::endl;
  } // end if
  std::cout << I.cols( ) << std::endl;

  /* TODO
     .select( I, 0 ).eval( );
     std::cout << J.rowwise( ).sum( ) << std::endl;
     std::cout << "_Z" << typeid( J ).name( ) << std::endl;
  */

  auto writer = itk::ImageFileWriter< TImage >::New( );
  writer->SetInput( out );
  writer->SetFileName( "labels.mha" );
  writer->Update( );

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
