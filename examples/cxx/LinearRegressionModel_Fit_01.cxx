// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/IO/CSV.h>
#include <PUJ_ML/Model/Regression/Linear.h>

int main( int argc, char** argv )
{
  PUJ_ML::Model::Regression::Linear< double > model;
  decltype( model )::TMatrix D;

  PUJ_ML::IO::CSV::read( D, argv[ 1 ] );

  model.fit(
    D.block( 0, 0, D.rows( ), D.cols( ) - 1 ),
    D.block( 0, D.cols( ) - 1, D.rows( ), 1 )
    );

  std::cout << "===============" << std::endl;
  std::cout << model << std::endl;
  std::cout << "===============" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
