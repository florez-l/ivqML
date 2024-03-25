// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <ivqML/Model/Regression/Linear.h>
#include <ivqML/Cost/MeanSquareError.h>
#include <ivqML/Optimizer/ADAM.h>
#include <ivqML/Trainer/CommandLine.h>

#include <boost/program_options.hpp>

using TReal  = double;
using TModel = ivqML::Model::Regression::Linear< TReal >;
using TCost  = ivqML::Cost::MeanSquareError< TModel >;
using TOptimizer = ivqML::Optimizer::ADAM< TCost >;

int main( int argc, char** argv )
{
  // Default arguments
  unsigned int N = 1;
  unsigned int M = 100;

  // Initial optimizer
  ivqML::Trainer::CommandLine< TOptimizer > optimizer;

  // Command line arguments
  namespace bpo = boost::program_options;
  bpo::options_description opt { "Options." };
  opt.add_options( )( "help,h", "help message" )
    ( "inputs", bpo::value< decltype( N ) >( &N )->default_value( N ) , "" )
    ( "samples", bpo::value< decltype( M ) >( &M )->default_value( M ) , "" );
  optimizer.register_options( opt );

  bpo::variables_map vm;
  bpo::store(
    bpo::command_line_parser( argc, argv ).options( opt )
    .allow_unregistered( ).run( ),
    vm
    );
  bpo::notify( vm );
  if( vm.count( "help" ) )
  {
    std::cerr << opt << std::endl;
    return( EXIT_FAILURE );
  } // end if

  // Model to generate data
  TModel original_model( N );
  original_model.random_fill( );
  std::cout << "Real model: " << original_model << std::endl;

  // Some random input data
  TModel::TMat X( N, M );
  X.setRandom( );
  TModel::TMat Y = original_model.eval( X );

  // Model to optimize
  TModel opt_model( N );
  opt_model.random_fill( );
  std::cout << "Initial model: " << opt_model << std::endl;

  optimizer.set_data( X, Y );
  optimizer.fit( opt_model );

  std::cout << "Fitted model: " << opt_model << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
