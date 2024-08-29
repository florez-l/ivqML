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

  std::vector< unsigned long long > idx( I.cols( ) );
  std::iota( idx.begin( ), idx.end( ), 0 );
  unsigned long long seed
    =
    std::chrono::system_clock::now( ).time_since_epoch( ).count( );
  std::shuffle( idx.begin( ), idx.end( ), std::default_random_engine( seed ) );

  using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
  TMatrix M[ 2 ];
  M[ 0 ] = I( { idx[ 0 ], idx[ 1 ], idx[ 2 ] }, ivq_EIGEN_ALL );
  M[ 1 ] = TMatrix::Zero( M[ 0 ].rows( ), M[ 0 ].cols( ) );
  unsigned long long K = M[ 0 ].rows( );

  std::cout << M[ 0 ] << std::endl;
  std::cout << K << std::endl;

  // Update distances
  TMatrix D( I.rows( ), K );
  for( unsigned long long k = 0; k < K; ++k )
  {
    D.col( k )
      =
      ( I.rowwise( ) - M[ 0 ].row( k ) ).array( ).
      pow( 2 ).rowwise( ).sum( ).sqrt( );
  } // end if

  // Update labels
  for( unsigned long long s = 0; s < L.rows( ); ++s )
    D.row( s ).minCoeff( &L( s, 0 ) );

  // Update means
  TMatrix J( I.rows( ), 1 );
  for( unsigned long long k = 0; k < K; ++k )
  {
    J = ( L.array( ) == k ).template cast< TReal >( );
    M[ 1 ].row( k )
      =
      ( I.array( ).colwise( ) * J.col( 0 ).array( ) ).colwise( ).sum( )
      /
      TReal( J.sum( ) );
  } // end for

  std::cout << "-------------------" << std::endl;
  std::cout << M[ 1 ] << std::endl;
  std::cout << "-------------------" << std::endl;
  std::cout << ( M[ 0 ] - M[ 1 ] ).array( ).pow( 2 ).rowwise( ).sum( ).mean( ) << std::endl;
  /* TODO


     for( unsigned long long l = 0; l < k; ++l )
     {
     auto J = ( L.array( ) == l ).template cast< unsigned long long >( ).eval( );
     std::cout << J.select( I, 0 ).rows( ) << " " << J.select( I, 0 ).cols( ) << std::endl;
     std::cout << "---------------------" << std::endl;
     } // end if
     std::cout << I.cols( ) << std::endl;
  */

  /* TODO
     .select( I, 0 ).eval( );
     std::cout << J.rowwise( ).sum( ) << std::endl;
     std::cout << "_Z" << typeid( J ).name( ) << std::endl;
  */

  /* TODO
     auto writer = itk::ImageFileWriter< TImage >::New( );
     writer->SetInput( out );
     writer->SetFileName( "labels.mha" );
     writer->Update( );
  */

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
