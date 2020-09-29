
#include "ActivationFunctions.h"
#include "NeuralNetwork.h"
#include "CSVReader.h"

#include <iomanip>
#include <iostream>

using TScalar = long double;
using TAnn = NeuralNetwork< TScalar >;

int main( int argc, char** argv )
{
  if( argc < 6 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ] << " train test validation alpha lambda" 
      << std::endl;
    return( 1 );
  } // end if

  std::string train_file = argv[ 1 ];
  std::string test_file = argv[ 2 ];
  std::string validation_file = argv[ 3 ];
  TScalar alpha = std::atof( argv[ 4 ] );
  TScalar lambda = std::atof( argv[ 5 ] );

  // Read data
  CSVReader train_reader( train_file, "," );
  train_reader.read( );
  TAnn::TMatrix X_train, Y_train;
  train_reader.cast( X_train, Y_train, 1 );

  CSVReader test_reader( test_file, "," );
  test_reader.read( );
  TAnn::TMatrix X_test, Y_test;
  test_reader.cast( X_test, Y_test, 1 );

  CSVReader validation_reader( validation_file, "," );
  validation_reader.read( );
  TAnn::TMatrix X_validation, Y_validation;
  validation_reader.cast( X_validation, Y_validation, 1 );

  // Normalization
  TAnn::TRowVector min_D = X_train.colwise( ).minCoeff( );
  TAnn::TRowVector max_D = X_train.colwise( ).maxCoeff( );
  TAnn::TRowVector dif_D = max_D - min_D;
  X_train.rowwise( ) -= min_D;
  X_train.array( ).rowwise( ) /= dif_D.array( );

  X_test.rowwise( ) -= min_D;
  X_test.array( ).rowwise( ) /= dif_D.array( );

  X_validation.rowwise( ) -= min_D;
  X_validation.array( ).rowwise( ) /= dif_D.array( );

  // Create
  TAnn ann( 5e-6 );
  ann.add( X_train.cols( ), 8, ActivationFunctions::ReLU< TScalar >( ) );
  ann.add( 1, ActivationFunctions::Logistic< TScalar >( ) );

  // Train
  ann.init( true );
  ann.train( X_train, Y_train, alpha, lambda, &std::cout );

  // Evaluate trained results
  TAnn::TMatrix K_train = ann.confusion_matrix( X_train, Y_train );
  std::cout
    << "****************************" << std::endl
    << "***** Training results *****" << std::endl
    << "****************************" << std::endl
    << "* Confusion matrix:" << std::endl << K_train << std::endl
    << std::setprecision( 4 )
    << "* Sen (0) : "
    << ( 100.0 * ( K_train( 0, 0 ) / ( K_train( 0, 0 ) + K_train( 1, 0 ) ) ) )
    << "%" << std::endl
    << "* PPV (0) : "
    << ( 100.0 * ( K_train( 0, 0 ) / ( K_train( 0, 0 ) + K_train( 0, 1 ) ) ) )
    << "%" << std::endl
    << "* Spe (1) : "
    << ( 100.0 * ( K_train( 1, 1 ) / ( K_train( 1, 1 ) + K_train( 0, 1 ) ) ) )
    << "%" << std::endl
    << "* NPV (1) : "
    << ( 100.0 * ( K_train( 1, 1 ) / ( K_train( 1, 1 ) + K_train( 1, 0 ) ) ) )
    << "%" << std::endl
    << "* F1      : "
    << ( ( 2.0 * K_train( 0, 0 ) ) / ( ( 2.0 * K_train( 0, 0 ) ) + K_train( 0, 1 ) + K_train( 1, 0 ) ) )
    << std::endl
    << "*******************" << std::endl;

  TAnn::TMatrix K_test = ann.confusion_matrix( X_test, Y_test );
  std::cout
    << "****************************" << std::endl
    << "***** Testing results *****" << std::endl
    << "****************************" << std::endl
    << "* Confusion matrix:" << std::endl << K_test << std::endl
    << std::setprecision( 4 )
    << "* Sen (0) : "
    << ( 100.0 * ( K_test( 0, 0 ) / ( K_test( 0, 0 ) + K_test( 1, 0 ) ) ) )
    << "%" << std::endl
    << "* PPV (0) : "
    << ( 100.0 * ( K_test( 0, 0 ) / ( K_test( 0, 0 ) + K_test( 0, 1 ) ) ) )
    << "%" << std::endl
    << "* Spe (1) : "
    << ( 100.0 * ( K_test( 1, 1 ) / ( K_test( 1, 1 ) + K_test( 0, 1 ) ) ) )
    << "%" << std::endl
    << "* NPV (1) : "
    << ( 100.0 * ( K_test( 1, 1 ) / ( K_test( 1, 1 ) + K_test( 1, 0 ) ) ) )
    << "%" << std::endl
    << "* F1      : "
    << ( ( 2.0 * K_test( 0, 0 ) ) / ( ( 2.0 * K_test( 0, 0 ) ) + K_test( 0, 1 ) + K_test( 1, 0 ) ) )
    << std::endl
    << "*******************" << std::endl;

  return( 0 );
}

// eof - $RCSfile$
