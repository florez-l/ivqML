// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>

#include <boost/program_options.hpp>

#include <PUJ_ML/Helpers/CSV.h>
#include <PUJ_ML/Model/Linear.h>
#include <PUJ_ML/Optimizer/GradientDescent.h>

// -- Types
using TScalar = double;
using TModel = PUJ_ML::Model::Linear< TScalar >;
using TMatrix = TModel::TMatrix;
namespace po = boost::program_options;

// -- Main --
int main( int argc, char** argv )
{
  std::string csv = "[NO INPUT FILE]";
  TScalar alpha = 1e-2;
  TScalar lambda = 0;
  TScalar epsilon = std::numeric_limits< TScalar >::epsilon( );
  unsigned long long epochs = 10000;
  unsigned long long debug_step = 100;
  bool use_LASSO = false;

  po::options_description desc( "Allowed parameters" );
  desc.add_options( )
    ( "help,h", "Help message" )
    ( "alpha,a", po::value( &alpha )->default_value( alpha ), "Learning rate" )
    ( "lambda,l", po::value( &lambda )->default_value( lambda ), "Regularization" )
    ( "LASSO", po::bool_switch( &use_LASSO )->default_value( use_LASSO ), "Use LASSO?" )
    ( "epsilon,e", po::value( &epsilon )->default_value( epsilon ), "Epsilon" )
    ( "epochs", po::value( &epochs )->default_value( epochs ), "Epochs" )
    ( "debug_step", po::value( &debug_step )->default_value( debug_step ), "Debug step" )
    ( "csv", po::value( &csv )->default_value( csv ), "Input file" )
    ;

  po::variables_map vm;
  po::store( po::parse_command_line( argc, argv, desc ), vm );
  po::notify( vm );
  if( vm.count( "help" ) )
  {  
    std::cerr << desc << std::endl;
    return( EXIT_FAILURE );
  } // end if

  auto D = PUJ_ML::Helpers::CSV::Read< TMatrix >( csv, true, "," );
  unsigned long long p = 1;
  unsigned long long m = D.rows( );
  unsigned long long n = D.cols( ) - p;
  TMatrix X = D.block( 0, 0, m, n );
  TMatrix Y = D.block( 0, n, m, p );

  TModel model;
  model.SetParameters( TModel::TCol::Zero( n + 1 ) );
  TModel::Cost cost( &model, X, Y );

  PUJ_ML::Optimizer::GradientDescent< TModel > opt;
  opt.SetCost( cost );
  opt.SetLearningRate( alpha );
  opt.SetRegularizationCoefficient( lambda );
  if( use_LASSO ) opt.SetRegularizationToLASSO( );
  else            opt.SetRegularizationToRidge( );
  opt.SetEpsilon( epsilon );
  opt.SetNumberOfEpochs( epochs );
  opt.SetDebugStep( debug_step );

  TScalar final_cost;
  unsigned long long final_epochs;
  opt.SetDebug(
    [&]( unsigned long long i, TScalar J, bool show ) -> bool
    {
      if( show )
        std::cout << i << " " << J << std::endl;
      final_cost = J;
      final_epochs = i;
      return( false );
    }
    );
  opt.Fit( );

  std::cout << "---------------------" << std::endl;
  std::cout << "Fitted model : " << model << std::endl;
  std::cout << "Final cost   : " << final_cost << std::endl;
  std::cout << "Final epochs : " << final_epochs << std::endl;
  std::cout << "---------------------" << std::endl;

  /* TODO
     std::cout << X.colwise( ).minCoeff( ) << std::endl;
     std::cout << X.colwise( ).maxCoeff( ) << std::endl;
     TModel::TCol L = TModel::TCol::LinSpaced( 100, -10, 10 );
  */

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
