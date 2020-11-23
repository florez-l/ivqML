// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "NeuralNetwork.h"
#include "ClassificationTrainer.h"
#include "CSVReader.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>

// -- Some typedefs
using TPixel = unsigned short;
using TScalar = double;
using TAnn = NeuralNetwork< TScalar >;
using TTrainer = ClassificationTrainer< TAnn >;

// -- Main function
int main( int argc, char** argv )
{
  if( argc < 7 )
  {
    std::cerr
      << "Usage: "
      << argv[ 0 ]
      << " input_examples.csv train test y_size alpha lambda" 
      << std::endl;
    return( 1 );
  } // end if
  std::string input_examples = argv[ 1 ];
  TScalar train_size = std::atof( argv[ 2 ] );
  TScalar test_size = std::atof( argv[ 3 ] );
  unsigned long p = std::atoi( argv[ 4 ] );
  TScalar alpha = std::atof( argv[ 5 ] );
  TScalar lambda = std::atof( argv[ 6 ] );

  // Read data
  CSVReader reader( input_examples, "," );
  reader.read( );
  TAnn::TMatrix X, Y;
  reader.cast( X, Y, p );

  // Create an empty artifical neural network
  TAnn ann;
  ann.add( X.cols( ), X.cols( ) * 4, "relu" );
  ann.add( X.cols( ) * 3, "relu" );
  ann.add( X.cols( ) * 2, "relu" );
  ann.add( p, "logistic" );

  // Train the neural network
  TTrainer tr( &ann );
  tr.setData( X.transpose( ), Y.transpose( ) );
  tr.setEpsilon( std::numeric_limits< float >::epsilon( ) );
  tr.setLearningRate( alpha );
  tr.setRegularization( lambda );
  tr.setBatchSize( 0 );
  tr.setSizes( train_size, test_size );
  tr.setNormalizationToStandardization( );
  ann.init( );
  tr.train(
    [&]( const unsigned long& i, const TScalar& Jtrain, const TScalar& Jtest )
    {
      if( i % 1000 == 0 )
        std::cout
          << std::scientific << std::setprecision( 4 )
          << "\33[2K\rIteration: " << i
          << "\tJtrain = " << Jtrain
          << "\tJtest = " << Jtest
          << " (epsilon = " << tr.epsilon( ) << ")"
          << std::flush;
    }
    );
  std::cout << std::endl << "done!" << std::endl;
  std::cout
    << std::fixed << std::setprecision( 4 )
    << "*** Train F1      = " << tr.FtrainScore( ) << " ***" << std::endl
    << "*** Test F1       = " << tr.FtestScore( ) << " ***" << std::endl
    << "*** Validation F1 = " << tr.FvalidScore( ) << " ***" << std::endl;

  return( 0 );
}

// eof - $RCSfile$
