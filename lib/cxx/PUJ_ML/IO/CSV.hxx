// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__IO__CSV__hxx__
#define __PUJ_ML__IO__CSV__hxx__

#include <iostream>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>

// -------------------------------------------------------------------------
template< class _X >
bool PUJ_ML::IO::CSV::
read(
  Eigen::EigenBase< _X >& X, const std::string& fname,
  unsigned int ignored_rows, const char& separator
  )
{
  std::ifstream in_f( fname.c_str( ) );
  in_f.seekg( 0, std::ios::end );
  size_t size = in_f.tellg( );
  std::string buffer( size, ' ' );
  in_f.seekg( 0 );
  in_f.read( &( buffer[ 0 ] ), size );
  in_f.close( );
  std::istringstream in_s( buffer );

  std::vector< typename _X::Scalar > data;
  std::string line;
  unsigned int rows = 0;
  while( std::getline( in_s, line ) )
  {
    if( ignored_rows <= rows )
    {
      std::replace( line.begin( ), line.end( ), separator, ' ' );
      std::istringstream in_l( line );
      while( !( in_l.eof( ) ) )
      {
        typename _X::Scalar v;
        in_l >> v;
        data.push_back( v );
      } // end while
    } // end if
    rows++;
  } // end while
  X.derived( ) = Eigen::Map< Eigen::Matrix< typename _X::Scalar, Eigen::Dynamic, Eigen::Dynamic > >( data.data( ), ( data.size( ) / rows ), rows ).transpose( );

  return( true );
}

#endif // __PUJ_ML__IO__CSV__hxx__

// eof - $RCSfile$
