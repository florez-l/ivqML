// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "CSVReader.h"
#include <fstream>
#include <regex>
#include <sstream>
#include <boost/algorithm/string.hpp>

// -------------------------------------------------------------------------
CSVReader::
CSVReader( const std::string& fname, const std::string& delimiters )
  : m_FileName( fname ),
    m_Delimiters( delimiters )
{
}

// -------------------------------------------------------------------------
void CSVReader::
read( )
{
  this->m_Data.clear( );

  std::ifstream in( this->m_FileName );
  std::string line = "";
  while( std::getline( in, line ) )
  {
    std::vector< std::string > tokens;
    boost::algorithm::split(
      tokens, line, boost::is_any_of( this->m_Delimiters )
      );
    bool valid = true;
    for( const std::string& t: tokens )
      valid &= std::regex_match( t, std::regex( "-?[0-9]+([\\.][0-9]+)?" ) );
    if( valid )
      this->m_Data.push_back( tokens );
  } // end while
  in.close( );
}

// -------------------------------------------------------------------------
template< class _TMatrixX, class _TMatrixY >
void CSVReader::
cast( _TMatrixX& X, _TMatrixY& Y, const int& p ) const
{
  assert( this->m_Data.size( ) > 0 );

  long m = this->m_Data.size( );
  long n = this->m_Data[ 0 ].size( ) - p;

  X = _TMatrixX::Zero( m, n );
  Y = _TMatrixY::Zero( m, p );
  for( long i = 0; i < m; ++i )
  {
    // Example
    for( long j = 0; j < n; ++j )
    {
      std::istringstream t( this->m_Data[ i ][ j ] );
      t >> X( i, j );
    } // end for

    // Result
    for( long k = 0; k < p; ++k )
    {
      std::istringstream t( this->m_Data[ i ][ k + n ] );
      t >> Y( i, k );
    } // end for
  } // end for
}

// -------------------------------------------------------------------------
#define CSVReader_cast_eigen_dense( _t )                                \
  template void                                                         \
  CSVReader::cast<                                                      \
    Eigen::Matrix< _t, Eigen::Dynamic, Eigen::Dynamic >,                \
    Eigen::Matrix< _t, Eigen::Dynamic, Eigen::Dynamic > >(              \
      Eigen::Matrix< _t, Eigen::Dynamic, Eigen::Dynamic >&,             \
      Eigen::Matrix< _t, Eigen::Dynamic, Eigen::Dynamic >&,             \
      const int&                                                        \
      ) const

#include <Eigen/Core>
CSVReader_cast_eigen_dense( float );
CSVReader_cast_eigen_dense( double );
CSVReader_cast_eigen_dense( long double );

// eof - $RCSfile$
