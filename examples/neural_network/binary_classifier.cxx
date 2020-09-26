// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "ActivationFunctions.h"
#include "NeuralNetwork.h"
#include "CSVReader.h"
#include <fstream>
#include <iostream>
#include <sstream>

// -- Some typedefs
using TScalar = float;
using TAnn = NeuralNetwork< TScalar >;

// -- Main function
int main( int argc, char** argv )
{
  // Check inputs and get them
  if( argc < 6 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ] << " input_examples.csv y_size alpha lambda epsilon"
      << std::endl;
    return( 1 );
  } // end if
  std::string input_examples = argv[ 1 ];
  std::stringstream args;
  args << argv[ 2 ] << " " << argv[ 3 ] << " " << argv[ 4 ]<< " " << argv[ 5 ];
  std::istringstream iargs( args.str( ) );
  int p;
  TScalar alpha, lambda, epsilon;
  iargs >> p >> alpha >> lambda >> epsilon;

  // Read data
  CSVReader reader( input_examples, "," );
  reader.read( );
  TAnn::TMatrix X, Y;
  reader.cast( X, Y, p );

  // Create an empty artifical neural network
  TAnn ann;
  ann.add( X.cols( ), X.cols( ) * 4, ActivationFunctions::ReLU< TScalar >( ) );
  ann.add( X.cols( ) * 3, ActivationFunctions::ReLU< TScalar >( ) );
  ann.add( X.cols( ) * 2, ActivationFunctions::ReLU< TScalar >( ) );
  ann.add( p, ActivationFunctions::Logistic< TScalar >( ) );

  // Initialize the ANN with random weights and biases
  ann.init( true );

  // Train the neural network
  ann.train( X, Y, alpha, lambda, epsilon, &std::cout );

  // Show results
  if( X.cols( ) == 2 )
  {
    auto minX = X.colwise( ).minCoeff( );
    auto maxX = X.colwise( ).maxCoeff( );
    auto difX = maxX - minX;

    unsigned long samples = 100;
    std::vector< unsigned char > data( 3 * samples * samples );
    unsigned long k = 0;
    for( unsigned long j = 0; j < samples; ++j )
    {
      TScalar dj = difX( 0, 1 ) * TScalar( j ) / TScalar( samples );
      dj += minX( 0, 1 );
      for( unsigned long i = 0; i < samples; ++i )
      {
        TScalar di = difX( 0, 0 ) * TScalar( i ) / TScalar( samples );
        di += minX( 0, 0 );
        TAnn::TRowVector x( X.cols( ) );
        x << di, dj;
        data[ k     ] = ( ann( x )( 0, 0 ) >= 0.5 )? 255: 0;
        data[ k + 2 ] = ( ann( x )( 0, 0 ) < 0.5 )? 255: 0;
        k += 3;
      } // end for
    } // end for

    // Save a file
    std::ofstream out( "ann.ppm" );
    out
      << "P6" << std::endl
      << "# Result of a 2-class ANN" << std::endl
      << samples << " " << samples << std::endl
      << "255" << std::endl;
    out.write( reinterpret_cast< char* >( data.data( ) ), 3 * samples * samples );
    out.close( );

  } // end if

  // Test the forward-propagation with a one-filled vector
  return( 0 );
}

// eof
