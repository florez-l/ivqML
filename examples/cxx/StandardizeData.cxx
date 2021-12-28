// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ/Data/CSV.h>
#include <PUJ/Traits.h>

// -- Typedef
using TScalar = double;
using TMatrix = PUJ::Traits< TScalar >::TMatrix;

// -------------------------------------------------------------------------
int main( int argc, char** argv )
{
  std::string input_csv_data = argv[ 1 ];
  std::string output_csv_data = argv[ 2 ];
  unsigned int output_size = 0;
  if( argc > 3 )
    output_size = std::atoi( argv[ 3 ] );

  TMatrix data = PUJ::CSV::Read< TMatrix >( input_csv_data );
  TMatrix X = data.block( 0, 0, data.rows( ), data.cols( ) - output_size );

  auto mean_v = X.colwise( ).mean( );
  auto C = X.rowwise( ) - mean_v;
  auto std_v =
    ( ( C.transpose( ) * C ).diagonal( ) / ( X.rows( ) - 1 ) ).
    array( ).sqrt( ).transpose( );

  data.block( 0, 0, data.rows( ), data.cols( ) - output_size ) =
    ( X.rowwise( ) - mean_v ).array( ).rowwise( ) / std_v.array( );

  if( PUJ::CSV::Write( data, output_csv_data ) )
    return( EXIT_SUCCESS );
  else
  {
    std::cerr << "Error writing data." << std::endl;
    return( EXIT_FAILURE );
  } // end if
}

// eof - $RCSfile$
