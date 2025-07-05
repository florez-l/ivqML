// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <itkVectorImage.h>
#include <ivq/ITK/EigenUtils.h>
#include <ivq/ITK/ImageFileReader.h>
#include <ivqML/Common/MeanShift.h>

#include <map>
#include <vector>
#include <itkImageRegionConstIterator.h>

int main( int argc, char** argv )
{
  using TReal = double;
  using TNatural = unsigned long long;
  using TMeanShift = ivqML::Common::MeanShift< TReal, TNatural >;
  using TImage = itk::VectorImage< TReal, 2 >;

  TNatural bins = 100;

  auto reader = ivq::ITK::ImageFileReader< TImage >::New( );
  reader->SetFileName( argv[ 1 ] );
  reader->Update( );
  // Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic > I = ivq::ITK::ImageToMatrix( reader->GetOutput( ) )( { 1 }, Eigen::placeholders::all );
  auto I = ivq::ITK::ImageToMatrix( reader->GetOutput( ) );
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

  Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic > H( I.rows( ), Hmap.size( ) );
  Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic > F( 1, Hmap.size( ) );
  TNatural k = 0;
  for( const auto& v: Hmap )
  {
    TNatural i = 0;
    for( const auto& j: v.first )
      H( i++, k ) = TReal( j );
    F( 0, k++ ) = TReal( v.second );
  } // end for

  TMeanShift ms( I.data( ), I.rows( ), I.cols( ) /*, F.data( )*/ );
  std::vector< TReal > means;
  ms.GetMeans( std::back_inserter( means ) );

  std::cout << H.cols( ) << " " << F.cols( ) << " " << ( means.size( ) / I.rows( ) ) << std::endl;
  /* TODO
     std::cout << "--------------------------------" << std::endl;
     for( const auto& v: means )
     std::cout << v << std::endl;
  */

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
