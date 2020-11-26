// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <sstream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/SVD>

// -- Some typedefs
using TScalar = double;
using TMatrix = Eigen::Matrix< TScalar, Eigen::Dynamic, Eigen::Dynamic >;

// -- Helper functions
void read_file( TMatrix& X, const std::string& filename );

// -- Main function
int main( int argc, char** argv )
{
  // Check inputs and get them
  if( argc < 2 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ] << " input.bin"
      << std::endl;
    return( 1 );
  } // end if
  std::string input = argv[ 1 ];

  // Read data
  TMatrix X;
  std::cout << "Read... " << std::flush;
  read_file( X, input );
  std::cout << "done! " << X.rows( ) << " " << X.cols( ) << std::endl;

  // Compute PCA
  std::cout << "Mean... " << std::flush;
  auto m = X.colwise( ).mean( );
  std::cout << "done! " << m.rows( ) << " " << m.cols( ) << std::endl;
  std::cout << "Centering... " << std::flush;
  auto c = X.rowwise( ) - m;
  std::cout << "done! " << c.rows( ) << " " << c.cols( ) << std::endl;
  std::cout << "Covariance... " << std::flush;
  auto S = ( c.transpose( ) * c ) / TScalar( X.cols( ) );
  std::cout << "done! " << S.rows( ) << " " << S.cols( ) << std::endl;
  std::cout << "Eigen analysis... " << std::flush;
  auto svd = S.bdcSvd( Eigen::ComputeFullU | Eigen::ComputeFullV );
  std::cout << "done!" << std::endl;
  TMatrix eR = svd.matrixU( );
  TMatrix ev = svd.singularValues( );
  TScalar sv = ev.sum( );

  unsigned long i = 0;
  double s = ev( i, 0 ) / sv;
  while( i < ev.rows( ) && s <= 0.95 )
  {
    i++;
    s += ev( i, 0 ) / sv;
  } // end while
  TMatrix Xp = ( X * eR ).block( 0, 0, X.rows( ), i );
  std::cout << i << " " << Xp.rows( ) << " " << Xp.cols( ) << std::endl;

  unsigned long xrows, xcols;
  xrows = Xp.rows( );
  xcols = Xp.cols( );

  std::ofstream out = std::ofstream( "out.bin", std::ios::binary );
  out.write( ( char* )( &xrows ), sizeof( unsigned long ) );
  out.write( ( char* )( &xcols ), sizeof( unsigned long ) );
  out.write( ( char* )( Xp.data( ) ), sizeof( TScalar ) * xrows * xcols );
  out.close( );

  /* TODO
     this->m_EigenValues.SetSize( ps );
     for( unsigned int d = 0; d < ps; ++d )
     this->m_EigenValues[ d ] = ev( d, 0 );
  */

  return( 0 );
}

// -------------------------------------------------------------------------
void read_file( TMatrix& X, const std::string& filename )
{
  std::ifstream Xreader = std::ifstream( filename, std::ios::binary );
  unsigned long xrows, xcols;
  Xreader.read( ( char* )( &xrows ), sizeof( unsigned long ) );
  Xreader.read( ( char* )( &xcols ), sizeof( unsigned long ) );
  X = TMatrix::Zero( xrows, xcols );
  Xreader.read( ( char* )( X.data( ) ), sizeof( TScalar ) * xrows * xcols );
  Xreader.close( );
}

// eof - pca.cxx
