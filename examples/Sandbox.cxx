// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>
#include <ivq/eigen/Config.h>

/* TODO
   #include <functional>
   #include <limits>
   
   #include <cstring>
*/

int main( int argc, char** argv )
{
  std::cout << "Number of threads: " << Eigen::nbThreads( ) << std::endl;

  using TReal = long double;
  using TNatural = unsigned long long;

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
  using TColumn    = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;
  using TColumnMap = Eigen::Map< TColumn >;
  
  struct SShiftCmp
  {
    bool operator()( const TColumnMap& a, const TColumnMap& b ) const
    {
      return( std::lexicographical_compare( a.begin( ), a.end( ), b.begin( ), b.end( ) ) );
    }
  };
  
  std::vector< TReal > means;
  std::map< TColumnMap, TColumnMap, SShiftCmp > means_map;

  auto distance = []( const TColumnMap& a, const TColumn& b ) -> TReal
  {
    return( std::sqrt( ( a - b ).array( ).pow( 2 ).sum( ) ) );
  };
  auto kernel = []( const TColumnMap& a, const TColumnMap& b ) -> TReal
  {
    TReal e = std::sqrt( ( a - b ).array( ).pow( 2 ).sum( ) ) / TReal( 1.5 );
    return( std::exp( TReal( -0.5 ) * ( e * e ) ) );
  };

  for( TNatural i = 0; i < M; ++i )
  {
    TColumnMap xi( I + ( i * N ), N, 1 );
    auto mIt = means_map.lower_bound( xi );
    bool found = ( mIt != means_map.end( ) );
    if( found ); // TODO
    if( !found )
    {
      for( TNatural n = 0; n < N; ++n )
        means.push_back( xi( n ) );
      TColumnMap xip( means.data( ) + ( means.size( ) - N ), N, 1 );
      
      bool stop = false;
      while( !stop )
      {
        TColumn xis = TColumn::Zero( N, 1 );
        TReal W = 0;
        for( TNatural j = 0; j < M; ++j )
        {
          TColumnMap xj( I + ( j * N ), N, 1 );
          TReal w = kernel( xip, xj );
          xis += xj * w;
          W += w;
        } // end for
        if( W != TReal( 0 ) ) 
          xis /= W;
        else
          xis.fill( 0 );
        TReal d = distance( xip, xis );
        std::cout << d << " --> " << xis.transpose( ) << std::endl;
        xip = xis;
      } // end while
    } // end if
    
    
  } // end for
  
  std::cout << means.size( ) << std::endl;
  
  

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
