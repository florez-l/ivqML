// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Data__CSV__hxx__
#define __PUJ__Data__CSV__hxx__

#include <filesystem>
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <Eigen/Core>

// -------------------------------------------------------------------------
template< class _TMatrix >
_TMatrix PUJ::CSV::Read(
  const std::string& fname,
  bool ignore_first_row,
  const std::string& separator
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
  std::istringstream input( buffer );

  // Read line by line
  std::vector< std::stringstream > lines;
  std::string line;
  unsigned long long n = 0;
  while( std::getline( input, line ) )
  {
    if( line != "" )
    {
      std::vector< std::string > tokens;
      boost::split( tokens, line, boost::is_any_of( separator ) );
      unsigned int i = 0;
      lines.push_back( std::stringstream( ) );
      for( const std::string& t: tokens )
      {
        if( t != "" )
        {
          *lines.rbegin( ) << t << " ";
          i++;
        } // end if
      } // end for
      if( i > 0 )
        n = ( n < i )? i: n;
      else
        lines.pop_back( );
    } // end if
  } // end while

  // Pass to Eigen::Matrix
  unsigned long long m = lines.size( );
  if( ignore_first_row )
    m--;
  _TMatrix data = _TMatrix::Zero( m, n );
  for( unsigned long long r = 0; r < m; ++r )
    for( unsigned long long c = 0; c < n; ++c )
      lines[ r ] >> data( r, c );

  // Finish
  return( data );
}

// -------------------------------------------------------------------------
template< class _TMatrix >
bool PUJ::CSV::Write(
  const _TMatrix& data, const std::string& fname, const char& separator
  )
{
  Eigen::IOFormat f(
    Eigen::StreamPrecision, Eigen::DontAlignCols,
    ",", "\n", "", "", "", "\n"
    );
  std::ofstream ofs( fname.c_str( ) );
  ofs << data.format( f );
  ofs.close( );

  return( true );
}

#endif // __PUJ__Data__CSV__hxx__

// eof - $RCSfile$
