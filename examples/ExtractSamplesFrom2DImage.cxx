// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <sstream>
#include <string>

#include "Helpers.h"

#include <itkImage.h>
#include <ivq/ITK/ImageFileReader.h>
#include <ivqML/IO/CSV.h>

using _R = double;
using _I = itk::Image< _R, 2 >;
using _M = Eigen::Matrix< _R, Eigen::Dynamic, Eigen::Dynamic >;

int main( int argc, char** argv )
{
  std::string in_fname = argv[ 1 ];
  std::string out_fname = argv[ 4 ];
  unsigned long long samples, labels;
  std::istringstream( argv[ 2 ] ) >> samples;
  std::istringstream( argv[ 3 ] ) >> labels;

  // Get input data
  auto reader = ivq::ITK::ImageFileReader< _I >::New( );
  reader->SetFileName( in_fname );
  reader->NormalizeOn( );
  reader->Update( );

  // Save as matricial data
  _M D =
    ivqML::Helpers::extract_discrete_samples_from_image< _I, _M >(
      reader->GetOutput( ), samples, labels
      );
  ivqML::IO::CSV::Write( D, out_fname, ',' );

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
