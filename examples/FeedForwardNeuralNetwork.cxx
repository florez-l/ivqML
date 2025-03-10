// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <ivqML/Model/NeuralNetwork/FeedForward.h>

/* TODO
   #include <algorithm>
   #include <cctype>
   #include <sstream>
   #include <string>
   #include <boost/program_options.hpp>
*/
/* TODO
   #include <ivqML/Helpers/ConfussionMatrix.h>
   #include <ivqML/Helpers/SplitDatasetWithBinaryLabeling.h>
   #include <ivqML/IO/CSV.h>
   #include <ivqML/Optimizer/Adam.h>
   #include <ivqML/Optimizer/GradientDescent.h>
   #include <ivqML/Optimizer/Debug/Simple.h>
*/

using TNatural = unsigned long long;
using TReal = long double;

int main( int argc, char** argv )
{
  using TModel = ivqML::Model::NeuralNetwork::FeedForward< TReal, TNatural >;

  TModel m;
  m.set_input_layer( 2, 10, "relu" );
  m.add_layer( 5, "relu" );
  m.add_layer( 3, "relu" );
  m.add_layer( 1, "identity" );
  m.init( );

  std::cout << "---------------------------------------------" << std::endl;
  std::cout << "Model: " << std::endl << m << std::endl;
  std::cout << "---------------------------------------------" << std::endl;

  TNatural M = 7;
  TModel::TMat X = TModel::TMat::Ones( 7, m.input_size( ) );
  auto y = m( X );

  std::cout << "---------------------------------------------" << std::endl;
  std::cout << X << std::endl;
  std::cout << "---------------------------------------------" << std::endl;
  std::cout << y << std::endl;
  std::cout << "---------------------------------------------" << std::endl;
  std::cout << "_Z" << typeid( y ).name( ) << std::endl;
  std::cout << "---------------------------------------------" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
