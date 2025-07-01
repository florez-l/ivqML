// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <vector>
#include <ivq/eigen/Config.h>

/* TODO
   #include <functional>
   
   
   
*/

int main( int argc, char** argv )
{
  std::cout << "Number of threads: " << Eigen::nbThreads( ) << std::endl;

  using TReal = long double;
  using TNatural = unsigned long long;
  using TColumn    = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;
  using TColumnMap = Eigen::Map< TColumn >;

  TReal I[ ] =
    {
      1.0, 1.0,
      1.2, 1.1,
      1.0, 1.3,
      1.1, 0.9,
      5.0, 5.0,
      5.1, 5.2,
      5.3, 5.0,
      5.0, 5.1,
      0.5, 6.0,
      0.7, 6.1,
      0.6, 5.9
    };
  TNatural M = 11;
  TNatural N = 2;
  
  TReal* O = reinterpret_cast< TReal* >( std::calloc( M * N, sizeof( TReal ) ) );
  
  struct SShiftCmp
  {
    bool operator()( const TColumnMap& a, const TColumnMap& b ) const
    {
      return( std::lexicographical_compare( a.begin( ), a.end( ), b.begin( ), b.end( ) ) );
    }
  };
  
  std::map< TColumnMap, TColumnMap, SShiftCmp > means_map;
  std::map< TColumnMap, std::vector< TColumnMap >, SShiftCmp > shifted_means_map;
  TReal dErr = std::pow( 10, std::log10( std::numeric_limits< TReal >::epsilon( ) ) * 0.5 );
  TNatural max_iter = 100;

  auto kernel = []( const TColumnMap& a, const TColumnMap& b ) -> TReal
  {
    TReal e = std::sqrt( ( a - b ).array( ).pow( 2 ).sum( ) ) / 1.5;
    return( std::exp( -0.5 * ( e * e ) ) );
  };

  for( TNatural i = 0; i < M; ++i )
  {
    TColumnMap xi( I + ( i * N ), N, 1 );
    TColumnMap xo( O + ( i * N ), N, 1 );
    xo = xi;
    
    auto mIt = means_map.end( );
    bool found = ( means_map.size( ) > 0 );
    if( found )
    {
      mIt = means_map.lower_bound( xi );
      if( mIt != means_map.end( ) );
    if( found )
      found = !( dErr < std::sqrt( ( xi - mIt->first ).array( ).pow( 2 ).sum( ) ) );
    } // end if
    if( !found )
    {
      bool stop = false;
      TColumn xs( N, 1 );
      TNatural k = 0;
      while( !stop )
      {
        xs.fill( 0 );
        TReal W = 0;
        for( TNatural j = 0; j < M; ++j )
        {
          TColumnMap xj( I + ( j * N ), N, 1 );
          TReal w = kernel( xo, xj );
          xs += xj * w;
          W += w;
        } // end for
        if( W != TReal( 0 ) ) 
          xs /= W;
        else
          xs.fill( 0 );
        TReal d = std::sqrt( ( xo - xs ).array( ).pow( 2 ).sum( ) );
        stop = !( dErr < d ) || !( ++k < max_iter );
        xo = xs;
        //if( stop )
        //std::cerr << k << " : " << d << " -> " << xo.transpose( ) << std::endl;
      } // end while
      auto smIt = shifted_means_map.lower_bound( xo );
      if( smIt != shifted_means_map.end( ) )
      {
        TReal d = std::sqrt( ( xo - smIt->first ).array( ).pow( 2 ).sum( ) );
        std::cout << "Distance: " << dErr << "|" << d << " <-> " << xo.transpose( ) << " *** " << smIt->first.transpose( ) << std::endl;
        if( dErr < d )
        {
          std::cout << "New2" << std::endl;
          means_map.insert( std::make_pair( xi, xo ) );
          shifted_means_map.insert( std::make_pair( xo, std::vector< TColumnMap >( ) ) ).first->second.push_back( xi );
        }
        else
        {
          means_map.insert( std::make_pair( xi, smIt->first ) );
          smIt->second.push_back( xi );
        } // end if
      }
      else
      {
        std::cout << "New" << std::endl;
        means_map.insert( std::make_pair( xi, xo ) );
        shifted_means_map.insert( std::make_pair( xo, std::vector< TColumnMap >( ) ) ).first->second.push_back( xi );
      } // end if
    }
    else
    {
      means_map.insert( std::make_pair( xi, mIt->second ) );
    } // end if
    
    
  } // end for
  
  std::cout << shifted_means_map.size( ) << std::endl;
  std::cout << "---------------------------------" << std::endl;
  for( auto s: shifted_means_map )
    std::cout << s.first.transpose( ) << " : " << s.second.size( ) << std::endl;

  std::free( O );

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
