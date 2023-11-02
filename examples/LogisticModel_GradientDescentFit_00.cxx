// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>

#include "Helpers.h"

#include <itkImage.h>
#include <ivq/ITK/ImageFileReader.h>

#include <ivqML/Model/Logistic.h>
#include <ivqML/Cost/CrossEntropy.h>
#include <ivqML/Optimizer/GradientDescent.h>

using _R = long double;
using _M = ivqML::Model::Logistic< _R >;
using _I = itk::Image< _R, 2 >;

int main( int argc, char** argv )
{
  // Get input data
  auto reader = ivq::ITK::ImageFileReader< _I >::New( );
  reader->SetFileName( argv[ 1 ] );
  reader->NormalizeOn( );
  reader->Update( );
  auto D =
    ivqML::Helpers::extract_discrete_samples_from_image< _I, _M::TMatrix >(
      reader->GetOutput( ), 5, 2
      );
  _M::TMatrix X = D.block( 0, 0, D.rows( ), D.cols( ) - 1 );
  _M::TMatrix Y = D.block( 0, D.cols( ) - 1, D.rows( ), 1 );

  // Model to be fitted
  _M fitted_model( 2 );
  fitted_model.random_fill( );
  std::cout << "Initial model : " << fitted_model << std::endl;

  // Optimization algorithm
  using _C = ivqML::Cost::CrossEntropy< _M >;
  ivqML::Optimizer::GradientDescent< _C > opt( fitted_model, X, Y );
  opt.set_debug(
    []( const _R& J, const _R& G, const _M* m, const _M::TNatural& i )
    -> bool
    {
      std::cerr << "J=" << J << ", Gn=" << G << ", i=" << i << std::endl;
      return( false );
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
  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
