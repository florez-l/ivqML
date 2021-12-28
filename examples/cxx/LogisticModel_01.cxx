// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <cmath>
#include <iostream>

#include <PUJ/Data/Algorithms.h>
#include <PUJ/Data/CSV.h>
#include <PUJ/Model/Logistic.h>
#include <PUJ/Optimizer/GradientDescent.h>
#include <PUJ/Optimizer/Trainer.h>

// -- Typedef
using TScalar    = double;
using TModel     = PUJ::Model::Logistic< TScalar >;
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
  unsigned long long m = X_real.rows( );
  unsigned long long n = X_real.cols( );

  // Configure trainer
  trainer.SetTrainData( X_real, y_real, PUJ::Zeros );
  trainer.SetDebug( debug );
  trainer.Fit( );

  // Accuracy
  TMatrix Y_real( m, 2 );
  Y_real.block( 0, 0, m, 1 ) =
    ( y_real.array( ) >= 0.5 ).template cast< TScalar >( );
  Y_real.block( 0, 1, m, 1 ) = 1 - Y_real.block( 0, 0, m, 1 ).array( );

  TMatrix Y_estim( m, 2 );
  Y_estim.block( 0, 0, m, 1 ) =
    ( trainer.GetModel( )( X_real ).array( ) >= 0.5 ).
    template cast< TScalar >( );
  Y_estim.block( 0, 1, m, 1 ) = 1 - Y_estim.block( 0, 0, m, 1 ).array( );

  TMatrix K = Y_real.transpose( ) * Y_estim;
  TScalar acc = TScalar( 100 ) * K.diagonal( ).sum( ) / K.sum( );

  // Show data
  std::cout << "=======================================" << std::endl;
  std::cout << "Optimized model  : " << trainer.GetModel( ) << std::endl;
  std::cout << "Iterations       : "
            << trainer.GetOptimizer( ).GetIterations( ) << std::endl;
  std::cout << "Accuracy         : " << acc << "%" << std::endl;
  std::cout << "Confusion matrix : " << std::endl;
  std::cout << K << std::endl;
  std::cout << "=======================================" << std::endl;
  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
