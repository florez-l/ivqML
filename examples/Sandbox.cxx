// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <functional>
#include <limits>
#include <map>
#include <vector>
#include <ivq/eigen/Config.h>

template< class _TReal, class _TNatural = unsigned long long >
class MeanShiftFunctions
{
public:
  using TReal = _TReal;
  using TNatural = _TNatural;

  using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
  using TColumn = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;

  using TMatrixMap = Eigen::Map< const TMatrix >;
  using TColumnMap = Eigen::Map< const TColumn >;

  using TDistance = std::function< TReal( const TColumnMap&, const TColumnMap& ) >;
  using TKernel = std::function< TReal( const TReal& ) >;

protected:
  struct SMapColCmp
  {
    bool operator()( const TColumnMap& a, const TColumnMap& b ) const
      {
        return(
          std::lexicographical_compare(
            a.data( ), a.data( ) + a.size( ),
            b.data( ), b.data( ) + b.size( )
            )
          );
      }
  };
  using TColumnRelation = std::map< TColumnMap, TColumnMap, SMapColCmp >;

public:
  MeanShiftFunctions( )
    {
      TReal b = std::log10( std::numeric_limits< TReal >::epsilon( ) );
      this->m_Epsilon = std::pow( TReal( 10 ), b * TReal( 0.5 ) );
      this->m_ConvergenceThreshold = std::pow( TReal( 10 ), b * TReal( 0.25 ) );

      this->m_Distance
        =
        []( const TColumnMap& a, const TColumnMap& b ) -> TReal
        {
          return( std::sqrt( ( a - b ).array( ).pow( 2 ).sum( ) ) );
        };
      this->m_Kernel
        =
        []( const TReal& d ) -> TReal
        {
          return( std::exp( ( d * d ) / TReal( -9 ) ) ); // ( -0.5 / 1.5 )
        };
    }
  virtual ~MeanShiftFunctions( )
    {
    }

  void shift( const TReal* I, const TNatural& N, const TNatural& M )
    {
      TMatrixMap D( I, N, M );
      TColumnRelation R;
      std::vector< TReal > S;

      for( TNatural c = 0; c < D.cols( ); ++c )
      {
        TColumnMap x( D.col( c ).data( ), D.rows( ), 1 );
        if( R.find( x ) == R.end( ) )
        {
          std::cout << ( c + 1 ) << " / " << D.cols( ) << " ---> (" << x.transpose( ) << ")" << std::endl;
          // SingleSampleMeanShift( std::back_inserter( shifted_means ), I, m, N, M, distance, kernel );
          // shifted_means_relations.insert( std::make_pair( x, shifted_means.size( ) / N ) );
        }
        else
        {
          std::cout << "uhhuoo" << std::endl;
        } // end if
      } // end for
    }

protected:
  TDistance m_Distance;
  TKernel   m_Kernel;

  TReal    m_Epsilon;
  TReal    m_ConvergenceThreshold;
  TNatural m_MaximumNumberOfIterations { 100 };
  TReal    m_ClusterMergeThreshold { TReal( 1 ) };
};

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
  TNatural N = 2;
  TNatural M = 11;

  MeanShiftFunctions< TReal, TNatural > ms_funcs;
  ms_funcs.shift( I, N, M );


  // Parameters for the Mean Shift algorithm
    /*
  double bandwidth = 1.5;            // Defines the radius of the search window. Crucial for results.
  double convergence_threshold = 0.001; // How small a shift is considered convergence.
  int max_iterations = 100;          // Maximum iterations for a single point's shift.
  double cluster_merge_threshold = 1.0; // Distance to merge two converged modes into one cluster.

  std::cout << "--- Mean Shift Clustering Example ---" << std::endl;
  std::cout << "Bandwidth: " << bandwidth << std::endl;
  std::cout << "Convergence Threshold: " << convergence_threshold << std::endl;
  std::cout << "Cluster Merge Threshold: " << cluster_merge_threshold << std::endl;
  std::cout << "Max Iterations per point: " << max_iterations << std::endl;
  std::cout << "-----------------------------------" << std::endl;
  
  auto distance
    =
    []( const TReal* a, const TReal* b, const TNatural& n ) -> TReal
    {
      TReal d = TReal( 0 );
      for( TNatural i = 0; i < n; ++i )
        d += ( a[ i ] - b[ i ] ) * ( a[ i ] - b[ i ] );
      return( std::sqrt( d ) );
    };
  TReal kernel_coeff = -TReal( 0.5 ) / ( bandwidth * bandwidth );
  TReal epsilon = std::numeric_limits< TReal >::epsilon( );
  auto kernel
    =
    [ &kernel_coeff, &epsilon ]( const TReal& d ) -> TReal
    {
      TReal w = std::exp( kernel_coeff * d * d );
      if( w <= epsilon )
        w = TReal( 0 );
      return( w );
    };

  MeanShift( I, N, M, distance, kernel );
*/
  /* TODO
     TMatrix I( 2, 11 );
     I.transpose( )
     <<
     1.0, 1.0, 1.2, 1.1, 1.0, 1.3, 1.1, 0.9, 5.0, 5.0,
     5.1, 5.2,
     5.3, 5.0, 5.0, 5.1, 0.5, 6.0, 0.7, 6.1, 0.6, 5.9;


     // Perform the clustering
     meanShiftClustering(
     I, bandwidth, convergence_threshold, max_iterations, cluster_merge_threshold
     );
  */

  /* TODO
     std::vector<Point> final_centroids = meanShiftClustering(data, bandwidth,
     convergence_threshold,
     max_iterations,
     cluster_merge_threshold);
     
     std::cout << "\n--- Clustering Results ---" << std::endl;
     for (size_t i = 0; i < data.size(); ++i) {
     std::cout << "Original Point (" << data[i].coords[0] << ", " << data[i].coords[1]
     << ") assigned to Cluster ID: " << data[i].cluster_id << std::endl;
     }

     std::cout << "\n--- Final Cluster Centroids ---" << std::endl;
     std::cout << "Found " << final_centroids.size() << " unique clusters." << std::endl;
     for (const auto& center : final_centroids) {
     std::cout << "Cluster " << center.cluster_id << " centroid: (";
     for (int d = 0; d < center.dimension; ++d) {
     std::cout << center.coords[d] << (d == center.dimension - 1 ? "" : ", ");
     }
     std::cout << ")" << std::endl;
     }
  */


  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
