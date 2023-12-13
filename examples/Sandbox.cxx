// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <fstream>
#include <ivqML/Config.h>
#include <ivqML/IO/CSV.h>

// -------------------------------------------------------------------------
int main( int argc, char** argv )
{
  std::cout << typeid( signed char ).name( ) << std::endl;
  std::cout << typeid( char ).name( ) << std::endl;
  std::cout << typeid( short ).name( ) << std::endl;
  std::cout << typeid( int ).name( ) << std::endl;
  std::cout << typeid( long ).name( ) << std::endl;
  std::cout << typeid( long long ).name( ) << std::endl;
  std::cout << typeid( unsigned char ).name( ) << std::endl;
  std::cout << typeid( unsigned short ).name( ) << std::endl;
  std::cout << typeid( unsigned int ).name( ) << std::endl;
  std::cout << typeid( unsigned long ).name( ) << std::endl;
  std::cout << typeid( unsigned long long ).name( ) << std::endl;
  std::cout << typeid( float ).name( ) << std::endl;
  std::cout << typeid( double ).name( ) << std::endl;
  std::cout << typeid( long double ).name( ) << std::endl;


  Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > D;
  ivqML::IO::CSV::Read( D, argv[ 1 ] );
  Eigen::Matrix< unsigned char, Eigen::Dynamic, Eigen::Dynamic > X;
  X = D.template cast< decltype( X )::Scalar >( );

  unsigned char i = typeid( unsigned char ).name( )[ 0 ];
  unsigned long long r = X.rows( );
  unsigned long long c = X.cols( );

  std::ofstream out( argv[ 2 ] );

  out.write( reinterpret_cast< char* >( &i ), sizeof( decltype( i ) ) );
  out.write( reinterpret_cast< char* >( &r ), sizeof( decltype( r ) ) );
  out.write( reinterpret_cast< char* >( &c ), sizeof( decltype( c ) ) );
  out.write(
    reinterpret_cast< char* >( X.data( ) ),
    sizeof( decltype( X )::Scalar ) * X.size( )
    );

  out.close( );


  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
