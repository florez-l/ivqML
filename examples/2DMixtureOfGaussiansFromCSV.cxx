// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include <ivqML/Common/MixtureOfGaussians.h>
#include <ivqML/IO/CSV.h>
#include <ivqML/VTK/Common/KMeans2DDebugger.h>

using TReal = float;
using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;

int main( int argc, char** argv )
{
  // Input arguments
  if( argc < 2 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ]
      << " input_csv [K=3] [init_method=\"xx\"]"
      << std::endl;
    return( EXIT_FAILURE );
  } // end if
  std::string input_csv = argv[ 1 ];
  unsigned int K = 3;
  std::string init_method = "++";
  if( argc > 2 ) std::istringstream( argv[ 2 ] ) >> K;
  if( argc > 3 ) init_method = argv[ 3 ];

  // Read input data
  TMatrix D;
  if( !ivqML::IO::CSV::Read( D, input_csv, 1, ',' ) )
  {
    std::cerr << "Error reading \"" << input_csv << "\"" << std::endl;
    return( EXIT_FAILURE );
  } // end if

  // Initialize problem with just two dimensions
  TMatrix X = D.block( 0, 0, D.rows( ), 2 );
  TMatrix means( K, X.cols( ) ), covariances;
  ivqML::Common::MixtureOfGaussians::Init( means, X, init_method );

  // Create visual debugger
  ivqML::VTK::Common::KMeans2DDebugger< TMatrix, TMatrix > debugger;
  debugger.Init( X, means );
  debugger.Start( );

  // Fit
  std::cout << "---- INIT ----" << std::endl;
  std::cout << means << std::endl;
  std::cout << "++++++++++++++" << std::endl;
  std::cout << covariances << std::endl;
  std::cout << "--------------" << std::endl;

  ivqML::Common::MixtureOfGaussians::Fit( means, covariances, X, debugger );

  std::cout << "---- FITTED ----" << std::endl;
  std::cout << means << std::endl;
  std::cout << "++++++++++++++++" << std::endl;
  std::cout << covariances << std::endl;
  std::cout << "----------------" << std::endl;

  debugger.Start( );

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
