// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__IO__Binary__hxx__
#define __ivqML__IO__Binary__hxx__

#include <filesystem>
#include <fstream>
#include <sstream>
#include <deque>
#include <boost/algorithm/string.hpp>
#include <Eigen/Core>

#define ivqML_IO_Binary_Read_Cast( _b, _t )                             \
  Eigen::Map< const Eigen::Matrix< _t, Eigen::Dynamic, Eigen::Dynamic > >( reinterpret_cast< const _t* >( _b ), r, c ).template cast< typename _M::Scalar >( )


// -------------------------------------------------------------------------
template< class _M >
bool ivqML::IO::Binary::Read(
  Eigen::EigenBase< _M >& M, const std::string& fname
  )
{
  // Load buffer
  std::ifstream ifs( fname.c_str( ) );
  ifs.seekg( 0, std::ios::end );
  std::size_t size = ifs.tellg( );
  ifs.seekg( 0, std::ios::beg );
  std::string buffer( size, 0 );
  ifs.read( &buffer[ 0 ], size );
  ifs.close( );

  // Base data
  unsigned char t = buffer[ 0 ];
  unsigned long long r =
    *( reinterpret_cast< const unsigned long long* >( buffer.data( ) + 1 ) );
  unsigned long long c =
    *( reinterpret_cast< const unsigned long long* >(
         buffer.data( ) + 1 + sizeof( unsigned long long )
         )
      );
  const char* b = buffer.data( ) + 1 + ( sizeof( unsigned long long ) << 1 );

  if( t == typeid( signed char ).name( )[ 0 ] )
    M.derived( ) = ivqML_IO_Binary_Read_Cast( b, signed char );
  else if( t == typeid( char ).name( )[ 0 ] )
    M.derived( ) = ivqML_IO_Binary_Read_Cast( b, char );
  else if( t == typeid( short ).name( )[ 0 ] )
    M.derived( ) = ivqML_IO_Binary_Read_Cast( b, short );
  else if( t == typeid( int ).name( )[ 0 ] )
    M.derived( ) = ivqML_IO_Binary_Read_Cast( b, int );
  else if( t == typeid( long ).name( )[ 0 ] )
    M.derived( ) = ivqML_IO_Binary_Read_Cast( b, long );
  else if( t == typeid( long long ).name( )[ 0 ] )
    M.derived( ) = ivqML_IO_Binary_Read_Cast( b, long long );
  else if( t == typeid( unsigned char ).name( )[ 0 ] )
    M.derived( ) = ivqML_IO_Binary_Read_Cast( b, unsigned char );
  else if( t == typeid( unsigned short ).name( )[ 0 ] )
    M.derived( ) = ivqML_IO_Binary_Read_Cast( b, unsigned short );
  else if( t == typeid( unsigned int ).name( )[ 0 ] )
    M.derived( ) = ivqML_IO_Binary_Read_Cast( b, unsigned int );
  else if( t == typeid( unsigned long ).name( )[ 0 ] )
    M.derived( ) = ivqML_IO_Binary_Read_Cast( b, unsigned long );
  else if( t == typeid( unsigned long long ).name( )[ 0 ] )
    M.derived( ) = ivqML_IO_Binary_Read_Cast( b, unsigned long long );
  else if( t == typeid( float ).name( )[ 0 ] )
    M.derived( ) = ivqML_IO_Binary_Read_Cast( b, float );
  else if( t == typeid( double ).name( )[ 0 ] )
    M.derived( ) = ivqML_IO_Binary_Read_Cast( b, double );
  else if( t == typeid( long double ).name( )[ 0 ] )
    M.derived( ) = ivqML_IO_Binary_Read_Cast( b, long double );
  else
    return( false );
  return( true );
}

// -------------------------------------------------------------------------
template< class _M >
bool ivqML::IO::Binary::Write(
  const Eigen::EigenBase< _M >& M, const std::string& fname
  )
{
  /* TODO
     std::stringstream seps;
     seps << separator;

     Eigen::IOFormat f(
     Eigen::StreamPrecision, Eigen::DontAlignCols,
     seps.str( ), "\n", "", "", "", "\n"
     );
     std::ofstream ofs( fname.c_str( ) );
     for( unsigned long long c = 0; c < M.cols( ) - 1; ++c )
     ofs << "x_" << c << separator;
     ofs << "y" << std::endl;
     ofs << M.derived( ).format( f );
     ofs.close( );
  */
  return( true );
}

#endif // __ivqML__IO__Binary__hxx__

// eof - $RCSfile$
