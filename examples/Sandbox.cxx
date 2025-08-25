// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <itkVectorImage.h>
#include <ivq/eigen/Histogram.h>
#include <ivq/ITK/EigenUtils.h>
#include <ivq/ITK/ImageFileReader.h>
#include <ivqML/Common/MeanShift.h>

/* TODO

   #include <map>
   #include <vector>
   #include <itkImageRegionConstIterator.h>
*/

int main( int argc, char** argv )
{
  using TReal = double;
  using TImage = itk::VectorImage< TReal, 2 >;

  if( argc < 2 )
  {
    std::cerr << "Usage: " << argv[ 0 ] << " input [bins=100]" << std::endl;
    return( EXIT_FAILURE );
  } // end if
  int bins = 100;
  if( argc > 2 ) std::istringstream( argv[ 2 ] ) >> bins;

  try
  {
    TReal I[ ] =
      {
        1.0, 1.0, 1.0,
        1.2, 1.1, 1.0,
        1.0, 1.3, 1.0,
        1.1, 0.9, 1.0,
        5.0, 5.0, 1.0,
        5.1, 5.2, 1.0,
        5.3, 5.0, 1.0,
        5.0, 5.1, 1.0,
        0.5, 6.0, 1.0,
        0.7, 6.1, 1.0,
        0.6, 5.9, 1.0
      };
    Eigen::Map< Eigen::Matrix< TReal, 3, 11 > > R( I );

    
    /* TODO
       auto reader = ::ivq::ITK::ImageFileReader< TImage >::New( );
       reader->SetFileName( argv[ 1 ] );
       reader->Update( );

       auto R = ivq::ITK::ImageToMatrix( reader->GetOutput( ) );
    */

    /* TODO
       using THisto = ivq::eigen::Histogram< TReal >;
       THisto::TMatrix H;
       THisto::multidimensional( H, R.transpose( ), bins );
    */

    using TMeanShift = ivqML::Common::MeanShift< decltype( R ) >;
    TMeanShift ms( R );
    ms.Compute( );

    /* TODO
       auto M = ivqML::Common::MeanShift<>::Histogram( R.transpose( ) H );
       std::cout << "_Z" << typeid( M ).name( ) << std::endl;
       std::cout << M << std::endl;
    */
  }
  catch( const std::exception& err )
  {
    std::cerr
      << "Error caught: \"" << err.what( ) << "\""
      << std::endl;
    return( EXIT_FAILURE );
  } // end try
  return( EXIT_SUCCESS );




  /* TODO
     using TNatural = unsigned long long;
     using TMeanShift = ivqML::Common::MeanShift< TReal, TNatural >;

     TNatural bins = 256;

     auto reader = ivq::ITK::ImageFileReader< TImage >::New( );
     reader->SetFileName( argv[ 1 ] );
     reader->Update( );
     auto R = ivq::ITK::ImageToMatrix( reader->GetOutput( ) );
     TMatrix G( 1, 3 );
     G << 0.299, 0.587, 0.114;
     TMatrix I = G * R.block( 0, 0, 3, R.cols( ) );

     TReal min_I = I.minCoeff( );
     TReal max_I = I.maxCoeff( );

     struct HCmp
     {
     bool operator()( const std::vector< TNatural >& a, const std::vector< TNatural >& b ) const
     {
     return( std::lexicographical_compare( a.begin( ), a.end( ), b.begin( ), b.end( ) ) );
     }
     };

     std::map< std::vector< TNatural >, TNatural, HCmp > Hmap;
     for( TNatural c = 0; c < I.cols( ); ++c )
     {
     std::vector< TNatural > idx( I.rows( ) );
     idx.shrink_to_fit( );
     for( TNatural r = 0; r < I.rows( ); ++r )
     idx[ r ] = ( bins - 1 ) * ( I( r, c ) - min_I ) / ( max_I - min_I );
     Hmap.insert( std::make_pair( idx, 0 ) ).first->second += 1;
     } // end for

     TMatrix H( I.rows( ), Hmap.size( ) );
     TMatrix F( 1, Hmap.size( ) );
     TNatural k = 0;
     for( const auto& v: Hmap )
     {
     TNatural i = 0;
     for( const auto& j: v.first )
     H( i++, k ) = min_I + ( ( max_I - min_I ) * ( TReal( j ) / TReal( bins - 1 ) ) );
     F( 0, k++ ) = TReal( v.second );
     } // end for

     TMeanShift ms( H.data( ), H.rows( ), H.cols( ), F.data( ) );
     std::vector< TReal > means;
     ms.GetMeans( std::back_inserter( means ) );

     std::cout << H.cols( ) << " " << F.cols( ) << " " << ( means.size( ) / I.rows( ) ) << std::endl;
  */

  /* TODO
     std::cout << "--------------------------------" << std::endl;
     for( const auto& v: means )
     std::cout << v << std::endl;
  */

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
