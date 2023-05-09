// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/Model/Regression/Linear.h>

int main( int argc, char** argv )
{
  PUJ_ML::Model::Regression::Linear< double > model;
  model.set_number_of_parameters( 1 );
  model( 0 ) = 1;
  model( 1 ) = 3;

  decltype( model )::TMatrix X( 4, model.number_of_inputs( ) );
  X << 1, 2, 3, 4;
  decltype( model )::TCol Y;
  model.evaluate( Y, X );

  std::cout << "===============" << std::endl;
  std::cout << model << std::endl;
  std::cout << "===============" << std::endl;
  std::cout << Y << std::endl;
  std::cout << "===============" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
