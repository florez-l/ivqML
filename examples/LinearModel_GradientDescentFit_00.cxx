// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <csignal>
#include <iostream>

#include <ivqML/Model/Linear.h>
#include <ivqML/Cost/MSE.h>
#include <ivqML/Optimizer/GradientDescent.h>

using _R = long double;
using _M = ivqML::Model::Linear< _R >;

// -------------------------------------------------------------------------
bool manual_stop = false;
bool debug(
  const _R& J, const _R& G, const _M* m, const _M::TNatural& i, bool d
  )
{
  if( d )
    std::cout << "J=" << J << ", Gn=" << G << ", i=" << i << std::endl;
  return( manual_stop );
}

// -------------------------------------------------------------------------
int main( int argc, char** argv )
{
  // Detect ctrl-c event to stop optimization and finish training
  signal( SIGINT, []( int s ) -> void { manual_stop = true; } );

  unsigned int m = 600;

  // Model to generate data
  _M real_model( 1 );
  real_model[ 0 ] = 3;
  real_model[ 1 ] = -2.5;
  std::cout << "Real model    : " << real_model << std::endl;

  // Some random input data
  _M::TMatrix X( m, real_model.number_of_inputs( ) );
  X.setRandom( );
  X.array( ) *= 10;
  X.array( ) -= 5;

  _M::TMatrix Y( X.rows( ), 1 );
  real_model( Y, X );

  // Model to be fitted
  _M fitted_model = real_model;
  fitted_model.random_fill( );
  std::cout << "Initial model : " << fitted_model << std::endl;

  // Optimization algorithm
  using _C = ivqML::Cost::MSE< _M >;
  ivqML::Optimizer::GradientDescent< _C > opt( fitted_model, X, Y );
  opt.set_debug( debug );

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
