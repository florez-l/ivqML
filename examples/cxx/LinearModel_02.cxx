// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <cmath>
#include <iostream>

#include <PUJ/Data/Algorithms.h>
#include <PUJ/Data/CSV.h>
#include <PUJ/Model/Linear.h>
#include <PUJ/Optimizer/GradientDescent.h>
#include <PUJ/Optimizer/Trainer.h>

// -- Typedef
using TScalar    = double;
using TModel     = PUJ::Model::Linear< TScalar >;
using TOptimizer = PUJ::Optimizer::GradientDescent< TModel >;
using TTrainer   = PUJ::Optimizer::Trainer< TOptimizer >;
using TMatrix    = TModel::TMatrix;

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
  // Prepare experiment
  TTrainer trainer( "input_csv_data", std::string( "" ) );
  if( !trainer.ParseArguments( argc, argv ) )
    return( EXIT_FAILURE );
  std::string input_csv_data =
    trainer.GetParameter< std::string >( "input_csv_data" );

  // Read data
  TMatrix data = PUJ::CSV::Read< TMatrix >( input_csv_data );
  PUJ::Algorithms::Shuffle( data );
  TMatrix X_real = data.block( 0, 0, data.rows( ), data.cols( ) - 1 );
  TMatrix y_real = data.block( 0, data.cols( ) - 1, data.rows( ), 1 );

  // Configure trainer
  trainer.SetTrainData( X_real, y_real );
  trainer.SetDebug( debug );
  trainer.Fit( );

  // Analytical model
  /* TODO
  TModel a_model;
  a_model.AnalyticalFit( X_real, y_real );
*/

  // Optimized model
  /*
  TModel opt_model;
  opt_model.Init( X_real.cols( ), PUJ::Random );
  TModel::Cost J( &opt_model, X_real, y_real, bsize );
  TOptimizer opt( &J );

  if( opt.ParseArguments( argc, argv ) )
  {
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
  */
  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
