// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <string>
#include <sstream>
#include <ivqML/IO/CSV.h>
#include <ivqML/Model/Regression/Linear.h>

using TScalar = long double;
using TModel = ivqML::Model::Regression::Linear< TScalar >;

// -------------------------------------------------------------------------
int main( int argc, char** argv )
{
  if( argc < 2 )
  {
    std::cerr << "Usage: " << argv[ 0 ] << " csv [lambda]" << std::endl;
    return( EXIT_FAILURE );
  } // end if
  std::string csv = argv[ 1 ];
  TScalar lambda = 0;
  if( argc > 2 )
    std::istringstream( argv[ 2 ] ) >> lambda;

  TModel::TMatrix D;
  ivqML::IO::CSV::Read( D, csv );

  TModel model;
  model.fit(
    D.block( 0, 0, D.rows( ), D.cols( ) - 1 ).transpose( ),
    D.col( D.cols( ) - 1 ).transpose( ),
    lambda
    );

  std::cout << "Fitted model: " << model << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
