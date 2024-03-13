// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <ivqML/Model/Regression/Linear.h>
#include <ivqML/Cost/MeanSquareError.h>
#include <ivqML/Optimizer/GradientDescent.h>
#include <ivqML/Trainer/CommandLine.h>

using TReal  = long double;
using TModel = ivqML::Model::Regression::Linear< TReal >;
using TCost  = ivqML::Cost::MeanSquareError< TModel >;
using TOptimizer = ivqML::Optimizer::GradientDescent< TCost >;

int main( int argc, char** argv )
{
  // Optimizer
  ivqML::Trainer::CommandLine< TOptimizer > opt;
  std::string help = opt.parse_arguments( argc, argv );
  if( help != "" )
  {
    std::cerr << help << std::endl;
    return( EXIT_FAILURE );
  } // end if
  unsigned int samples = 100;

  // Model to generate data
  TModel original_model( 1 );
  original_model[ 0 ] = 3;
  original_model[ 1 ] = -2.5;
  std::cerr << "Real model: " << original_model << std::endl;

  // Some random input data
  unsigned int n = original_model.number_of_inputs( );
  unsigned int p = original_model.number_of_parameters( );
  TModel::TMat X( n, samples );
  X.setRandom( );
  X.array( ) *= 10;
  X.array( ) -= 5;
  TModel::TMat Y = original_model.eval( X );

  // Optimize!
  opt.set_data( X, Y );
  opt.fit( );

  // Show results
  std::cerr << "Final model: " << opt.model( ) << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
