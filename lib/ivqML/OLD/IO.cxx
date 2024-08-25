// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <fstream>
#include <sstream>
#include <vector>
#include <ivq/eigen/Config.h>
#include <ivq/eigen/IO.h>

// -------------------------------------------------------------------------
template< class _TMatrix >
void ivq::eigen::IO::readCSV(
  _TMatrix& X, const std::string& filename,
  int ignore_first_rows
  )
{
  std::ifstream ins( filename );
  std::stringstream inb;
  inb << ins.rdbuf( );
  ins.close( );

  // Read data line by line
  std::vector< typename _TMatrix::Scalar > data;
  std::string line;
  unsigned long long M = 0;
  long double v;
  while( std::getline( inb, line ) )
  {
    if( ignore_first_rows < 0 || M > ignore_first_rows )
    {
      std::istringstream iss( line );
      while( !iss.eof( ) )
      {
        iss >> v;
        data.push_back( ( typename _TMatrix::Scalar )( v ) );
      } // end while
    } // end if
    M++;
  } // end while
  if( ignore_first_rows >= 0 )
    M -= ignore_first_rows;

  // Load data
  unsigned long long N = data.size( ) / M;
  if( long( _TMatrix::Options ) == long( Eigen::ColMajor ) )
    X = Eigen::Map< _TMatrix >( data.data( ), N, M ).transpose( );
  else
    X = Eigen::Map< _TMatrix >( data.data( ), M, N );
}

// -------------------------------------------------------------------------
#define ivq_eigen_readCSV( _t_ )                                        \
  template void ivq_EXPORT ivq::eigen::IO::readCSV< Eigen::Matrix< _t_, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor > >( Eigen::Matrix< _t_, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor >&, const std::string&, int ); \
  template void ivq_EXPORT ivq::eigen::IO::readCSV< Eigen::Matrix< _t_, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > >( Eigen::Matrix< _t_, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor >&, const std::string&, int )

ivq_eigen_readCSV( char );
ivq_eigen_readCSV( short );
ivq_eigen_readCSV( int );
ivq_eigen_readCSV( long );
ivq_eigen_readCSV( long long );
ivq_eigen_readCSV( signed char );
ivq_eigen_readCSV( unsigned char );
ivq_eigen_readCSV( unsigned short );
ivq_eigen_readCSV( unsigned int );
ivq_eigen_readCSV( unsigned long );
ivq_eigen_readCSV( unsigned long long );
ivq_eigen_readCSV( float );
ivq_eigen_readCSV( double );
ivq_eigen_readCSV( long double );

// eof - $RCSfile$
