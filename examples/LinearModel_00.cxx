// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <chrono>
#include <iostream>
#include <ivqML/Model/Regression/Linear.h>
#include <ivqML/Cost/MeanSquareError.h>

using TReal = long double;
using TModel = ivqML::Model::Regression::Linear< TReal >;
using TCost = ivqML::Cost::MeanSquareError< TModel >;

int main( int argc, char** argv )
{
  auto ts = std::chrono::high_resolution_clock::now( );
  auto te = std::chrono::high_resolution_clock::now( );

  unsigned int n = 4;
  unsigned int m = 10;
  unsigned int r = 10;

  if( argc > 1 ) n = std::atoi( argv[ 1 ] );
  if( argc > 2 ) m = std::atoi( argv[ 2 ] );
  if( argc > 3 ) r = std::atoi( argv[ 3 ] );

  // A model
  TReal t = 0;
  TModel model( n );
  for( unsigned int i = 0; i < r; ++i )
  {
    ts = std::chrono::high_resolution_clock::now( );
    model.random_fill( );
    te = std::chrono::high_resolution_clock::now( );
    t +=
      TReal(
        std::chrono::duration_cast< std::chrono::nanoseconds >
        ( te - ts ).count( )
        ) / TReal( r );
  } // end for
  std::cout << "Model: " << model << " (" << t * 10e-9 << " s)" << std::endl;

  // Some random input data
  TModel::TMat X( n, m );
  X.setRandom( );

  // Evaluate from input data
  TModel::TMat Y;
  t = 0;
  for( unsigned int i = 0; i < r; ++i )
  {
    ts = std::chrono::high_resolution_clock::now( );
    Y = model.eval( X );
    te = std::chrono::high_resolution_clock::now( );
    t +=
      TReal(
        std::chrono::duration_cast< std::chrono::nanoseconds >
        ( te - ts ).count( )
        ) / TReal( r );
  } // end for
  std::cout << "Evaluate mean time: " << t * 10e-9 << " s" << std::endl;

  // New model for cost
  TModel model_for_cost( model.number_of_inputs( ) );
  model_for_cost.random_fill( );

  // Cost model
  TCost J;
  J.set_data( X.data( ), Y.data( ), m, n, 1 );
  TModel::TRow G( model_for_cost.number_of_parameters( ) );

  t = 0;
  for( unsigned int i = 0; i < r; ++i )
  {
    ts = std::chrono::high_resolution_clock::now( );
    J( model_for_cost, G.data( ) );
    te = std::chrono::high_resolution_clock::now( );
    t +=
      TReal(
        std::chrono::duration_cast< std::chrono::nanoseconds >
        ( te - ts ).count( )
        ) / TReal( r );
  } // end for
  std::cout << "Cost mean time: " << t * 10e-9 << " s" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
