// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <csignal>
#include <iostream>

#include "Helpers.h"

#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <ivq/ITK/ImageFileReader.h>

#include <ivqML/Model/Logistic.h>
#include <ivqML/Cost/CrossEntropy.h>
#include <ivqML/Optimizer/ADAM.h>

using _R = double;
using _M = ivqML::Model::Logistic< _R >;
using _I = itk::Image< _R, 2 >;

// Detect ctrl-c event to stop optimization and finish training
bool general_stop = false;
void sigint_handler( int s )
{
  general_stop = true;
}

int main( int argc, char** argv )
{
  // Get input data
  auto reader = ivq::ITK::ImageFileReader< _I >::New( );
  reader->SetFileName( argv[ 1 ] );
  reader->NormalizeOn( );
  reader->Update( );
  auto D =
    ivqML::Helpers::extract_discrete_samples_from_image< _I, _M::TMatrix >(
      reader->GetOutput( ), 500, 2
      );
  _M::TMatrix X = D.block( 0, 0, D.rows( ), D.cols( ) - 1 );
  _M::TMatrix Y = D.block( 0, D.cols( ) - 1, D.rows( ), 1 );

  // Model to be fitted
  _M fitted_model( 2 );
  fitted_model.random_fill( );
  std::cout << "Initial model : " << fitted_model << std::endl;

  // Optimization algorithm
  using _C = ivqML::Cost::CrossEntropy< _M >;
  ivqML::Optimizer::ADAM< _C > opt( fitted_model, X, Y );

  signal( SIGINT, sigint_handler );
  opt.set_debug(
    []( const _R& J, const _R& G, const _M* m, const _M::TNatural& i )
    -> bool
    {
      std::cerr << "J=" << J << ", Gn=" << G << ", i=" << i << std::endl;
      return( general_stop );
    }
    );

  std::string ret = opt.parse_options( argc, argv );
  if( ret != "" )
  {
    std::cerr << ret << std::endl;
    return( EXIT_FAILURE );
  } // end if
  opt.fit( );
  std::cout << "Fitted model  : " << fitted_model << std::endl;


  _I::Pointer output = _I::New( );
  output->CopyInformation( reader->GetOutput( ) );
  output->SetRequestedRegionToLargestPossibleRegion( );
  output->SetBufferedRegion( output->GetRequestedRegion( ) );
  output->Allocate( );

  auto roi = output->GetRequestedRegion( );
  unsigned long long Iw = roi.GetSize( )[ 0 ];
  unsigned long long Ih = roi.GetSize( )[ 1 ];
  unsigned long long Is = Iw * Ih;
  auto spac = output->GetSpacing( );
  auto orig = output->GetOrigin( );

  _M::TMatrix S = Eigen::Map< const Eigen::Matrix< typename decltype( spac )::ValueType, Eigen::Dynamic, Eigen::Dynamic > >( spac.data( ), 1, _I::ImageDimension ).template cast< _R >( );
  _M::TMatrix O = Eigen::Map< const Eigen::Matrix< typename decltype( orig )::ValueType, Eigen::Dynamic, Eigen::Dynamic > >( orig.data( ), 1, _I::ImageDimension ).template cast< _R >( );

  _M::TMatrix rows( Ih, 1 ), cols( Iw, 1 );
  rows.col( 0 ).setLinSpaced( Ih, 0, Ih - 1 );
  cols.col( 0 ).setLinSpaced( Iw, 0, Iw - 1 );
  _M::TMatrix Ix( Is, _I::ImageDimension );
  Ix
    <<
    cols.replicate( 1, Ih ).reshaped( Is, 1 ),
    rows.replicate( 1, Iw ).transpose( ).reshaped( Is, 1 );
  Ix.array( ).rowwise( ) *= S.array( ).row( 0 );
  Ix.array( ).rowwise( ) += O.array( ).row( 0 );

  auto Z = ivq::ITK::ImageToMatrix( output.GetPointer( ) ).transpose( );
  fitted_model( Z, Ix );

  auto writer = itk::ImageFileWriter< _I >::New( );
  writer->SetInput( output );
  writer->SetFileName( "out.mha" );
  writer->Update( );

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
