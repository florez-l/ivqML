// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/Helpers/CSV.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <deque>
#include <boost/algorithm/string.hpp>
#include <Eigen/Core>

// -------------------------------------------------------------------------
template< class _TMatrix >
_TMatrix PUJ_ML::Helpers::CSV::Read(
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
  std::deque< std::stringstream > lines;
  std::string line;
  unsigned long long n = 0;
  while( std::getline( input, line ) )
  {
    if( line != "" )
    {
      if( int( line[ 0 ] ) != 0 )
      {
        std::deque< std::string > tokens;
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
    } // end if
  } // end while

  // Pass to Eigen::Matrix
  if( ignore_first_row )
    lines.pop_front( );
  unsigned long long m = lines.size( );
  _TMatrix data = _TMatrix::Zero( m, n );
  for( unsigned long long r = 0; r < m; ++r )
    for( unsigned long long c = 0; c < n; ++c )
      lines[ r ] >> data( r, c );

  // Finish
  return( data );
}

// -------------------------------------------------------------------------
template< class _TMatrix >
bool PUJ_ML::Helpers::CSV::Write(
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

// -------------------------------------------------------------------------
#define PUJ_Helpers_CSV_Instance( _t_ )                                 \
  template Eigen::Matrix< _t_, Eigen::Dynamic, Eigen::Dynamic >         \
  PUJ_ML_EXPORT                                                         \
  PUJ_ML::Helpers::CSV::Read(                                           \
    const std::string&, bool, const std::string&                        \
    );                                                                  \
  template bool PUJ_ML_EXPORT                                           \
  PUJ_ML::Helpers::CSV::Write(                                          \
    const Eigen::Matrix< _t_, Eigen::Dynamic, Eigen::Dynamic >&,        \
    const std::string&, const char&                                     \
    );                                                                  \
  template bool PUJ_ML_EXPORT                                           \
  PUJ_ML::Helpers::CSV::Write(                                          \
    const Eigen::Matrix< _t_, 1, Eigen::Dynamic >&,                     \
    const std::string&, const char&                                     \
    );                                                                  \
  template bool PUJ_ML_EXPORT                                           \
  PUJ_ML::Helpers::CSV::Write(                                          \
    const Eigen::Matrix< _t_, Eigen::Dynamic, 1 >&,                     \
    const std::string&, const char&                                     \
    )

PUJ_Helpers_CSV_Instance( float );
PUJ_Helpers_CSV_Instance( double );
PUJ_Helpers_CSV_Instance( long double );

// eof - $RCSfile$
