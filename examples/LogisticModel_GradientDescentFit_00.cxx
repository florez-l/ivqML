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
#include <ivqML/Optimizer/GradientDescent.h>
#include <ivqML/ITK/ApplyModelToImageMeshFilter.h>

using _R = double;
using _M = ivqML::Model::Logistic< _R >;
using _I = itk::Image< _R, 2 >;
using _C = ivqML::Cost::CrossEntropy< _M >;
using _O = ivqML::Optimizer::GradientDescent< _C >;

// Detect ctrl-c event to stop optimization and finish training
bool general_stop = false;
void sigint_handler( int s )
{
  general_stop = true;
}

int main( int argc, char** argv )
{
  if( argc < 4 )
  {
    std::cerr << "Usage: " << argv[ 0 ] << " input output n" << std::endl;
    return( EXIT_FAILURE );
  } // end if

  // Get input data
  auto reader = ivq::ITK::ImageFileReader< _I >::New( );
  reader->SetFileName( argv[ 1 ] );
  reader->NormalizeOn( );
  reader->Update( );

  auto D =
    ivqML::Helpers::extract_discrete_samples_from_image< _I, _M::TMatrix >(
      reader->GetOutput( ), std::atoi( argv[ 3 ] ), 2
      );
  _M::TMatrix X = D.block( 0, 0, D.rows( ), D.cols( ) - 1 );
  _M::TMatrix Y = D.block( 0, D.cols( ) - 1, D.rows( ), 1 );

  // Model to be fitted
  _M fitted_model( 2 );
  fitted_model.random_fill( );
  std::cout << "Initial model : " << fitted_model << std::endl;

  // Optimization algorithm
  _O opt( fitted_model, X, Y );

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

  using _A = ivqML::ITK::ApplyModelToImageMeshFilter< _I, _M >;
  auto apply_model = _A::New( );
  apply_model->SetInput( reader->GetOutput( ) );
  apply_model->SetModel( fitted_model );

  auto writer = itk::ImageFileWriter< _A::TOutput >::New( );
  writer->SetInput( apply_model->GetOutput( ) );
  writer->SetFileName( argv[ 2 ] );
  writer->Update( );

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
