// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <cmath>
#include <iostream>

#include <PUJ/Data/Algorithms.h>
#include <PUJ/Data/CSV.h>
#include <PUJ/Model/Linear.h>
#include <PUJ/Optimizer/GradientDescent.h>

// -- Typedef
using TScalar    = double;
using TModel     = PUJ::Model::Linear< TScalar >;
using TOptimizer = PUJ::Optimizer::GradientDescent< TModel >;

// -------------------------------------------------------------------------
bool debug( unsigned long long i, TScalar J, TScalar dJ, bool show )
{
  if( show )
    std::cout
      << "Iteration: " << i
      << ",  Cost = " << J << " (" << dJ << ")" << std::endl;
  return( false );
}

// -------------------------------------------------------------------------
int main( int argc, char** argv )
{
  // Read data
  TModel::TMatrix data = PUJ::CSV::Read< TModel::TMatrix >( argv[ 1 ] );
  PUJ::Algorithms::Shuffle( data );
  TModel::TMatrix X_real = data.block( 0, 0, data.rows( ), data.cols( ) - 1 );
  TModel::TMatrix y_real = data.block( 0, data.cols( ) - 1, data.rows( ), 1 );

  // Analytical model
  TModel analytical_model;
  analytical_model.AnalyticalFit( X_real, y_real );

  // Optimized model
  TModel opt_model;
  opt_model.Init( X_real.cols( ), PUJ::Random );
  TModel::Cost J( &opt_model, X_real, y_real, 1 );
  TOptimizer opt( &J );

  if( opt.ParseArguments( argc, argv ) )
  {
    /* TODO
       opt.SetAlpha( 1e-4 );
       opt.SetMaximumNumberOfIterations( 100000 );
       opt.SetDebugIterations( 10 );
    */
    opt.SetDebug( debug );
    opt.Fit( );

    std::cout << "=======================================" << std::endl;
    std::cout << "Analytical model : " << analytical_model << std::endl;
    std::cout << "Optimized model  : " << opt_model << std::endl;
    std::cout << "Iterations       : " << opt.GetIterations( ) << std::endl;
    std::cout << "=======================================" << std::endl;

    return( EXIT_SUCCESS );
  }
  else
    return( EXIT_FAILURE );
}

// eof - $RCSfile$
