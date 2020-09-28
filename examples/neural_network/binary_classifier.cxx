// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "ActivationFunctions.h"
#include "NeuralNetwork.h"
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

// -- Main function
int main( int argc, char** argv )
{
  // Check inputs and get them
  if( argc < 5 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ] << " input_examples.csv y_size alpha lambda"
      << std::endl;
    return( 1 );
  } // end if
  std::string input_examples = argv[ 1 ];
  std::stringstream args;
  args << argv[ 2 ] << " " << argv[ 3 ] << " " << argv[ 4 ];
  std::istringstream iargs( args.str( ) );
  int p;
  TScalar alpha, lambda;
  iargs >> p >> alpha >> lambda;

  // Read data
  CSVReader reader( input_examples, "," );
  reader.read( );
  TAnn::TMatrix X, Y;
  reader.cast( X, Y, p );

  // Create an empty artifical neural network
  TAnn ann( 1e-6 );
  /* TODO
     ann.add( X.cols( ), X.cols( ) * 4, ActivationFunctions::ReLU< TScalar >( ) );
     ann.add( X.cols( ) * 3, ActivationFunctions::ReLU< TScalar >( ) );
     ann.add( X.cols( ) * 2, ActivationFunctions::ReLU< TScalar >( ) );
     ann.add( p, ActivationFunctions::Logistic< TScalar >( ) );
  */
  ann.add( X.cols( ), p, ActivationFunctions::Logistic< TScalar >( ) );

  // Initialize the ANN with random weights and biases
  ann.init( true );

  // Train the neural network
  ann.train( X, Y, alpha, lambda, &std::cout );

  // Evaluate trained results
  TAnn::TMatrix K = ann.confusion_matrix( X, Y );
  std::cout
    << "*******************" << std::endl
    << "***** Results *****" << std::endl
    << "*******************" << std::endl
    << "* Confusion matrix:" << std::endl << K << std::endl
    << std::setprecision( 4 )
    << "* Sen (0) : "
    << ( 100.0 * ( K( 0, 0 ) / ( K( 0, 0 ) + K( 1, 0 ) ) ) )
    << "%" << std::endl
    << "* PPV (0) : "
    << ( 100.0 * ( K( 0, 0 ) / ( K( 0, 0 ) + K( 0, 1 ) ) ) )
    << "%" << std::endl
    << "* Spe (1) : "
    << ( 100.0 * ( K( 1, 1 ) / ( K( 1, 1 ) + K( 0, 1 ) ) ) )
    << "%" << std::endl
    << "* NPV (1) : "
    << ( 100.0 * ( K( 1, 1 ) / ( K( 1, 1 ) + K( 1, 0 ) ) ) )
    << "%" << std::endl
    << "* F1      : "
    << ( ( 2.0 * K( 0, 0 ) ) / ( ( 2.0 * K( 0, 0 ) ) + K( 0, 1 ) + K( 1, 0 ) ) )
    << std::endl
    << "*******************" << std::endl;

  /* TODO
     if( X.cols( ) == 2 )
     {
     auto minX = X.colwise( ).minCoeff( );
     auto maxX = X.colwise( ).maxCoeff( );
     auto difX = maxX - minX;

     unsigned long samples = 100;
     std::vector< TPixel > data( 3 * samples * samples );
     unsigned long k = 0;
     for( unsigned long j = 0; j < samples; ++j )
     {
     TScalar dj = difX( 0, 1 ) * TScalar( j ) / TScalar( samples );
     dj += minX( 0, 1 );
     for( unsigned long i = 0; i < samples; ++i )
     {
     TScalar di = difX( 0, 0 ) * TScalar( i ) / TScalar( samples );
     di += minX( 0, 0 );
     TAnn::TColVector x( X.cols( ) );
     x << di, dj;
     data[ k ] =
     TPixel(
     ann( x )( 0, 0 ) *
     TScalar( std::numeric_limits< TPixel >::max( ) )
     );
     data[ k + 1 ] = data[ k + 2 ] = data[ k ];
     k += 3;
     } // end for
     } // end for

     // Save a file
     std::ofstream out( "ann.ppm" );
     out
     << "P6" << std::endl
     << "# Result of a 2-class ANN" << std::endl
     << samples << " " << samples << std::endl
     << std::numeric_limits< TPixel >::max( ) << std::endl;
     out.write( reinterpret_cast< char* >( data.data( ) ), 3 * samples * samples );
     out.close( );
     } // end if
  */

  return( 0 );
}

// eof
