// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <functional>
#include <limits>
#include <map>
#include <vector>
#include <cstring>
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
  MeanShiftFunctions( const TReal* I, const TNatural& N, const TNatural& M )
    {
      this->_init( );
      new ( &( this->m_Data ) ) TMatrixMap( I, N, M );
    }
  virtual ~MeanShiftFunctions( )
    {
      /* TODO
         if( this->m_Means != nullptr )
         std::free( this->m_Means );
      */
    }

  void compute_means( )
    {
      std::cout << this->m_Data << std::endl;

      // Prepare outputs
      /* TODO
         if( this->m_Means != nullptr )
         std::free( this->m_Means );
         this->m_Means = reinterpret_cast< TReal* >( std::calloc( N * M, sizeof( TReal ) ) );
         this->m_Relation.clear( );

         // Main loop
         const TReal* bI = I;
         TReal* bM = this->m_Means;
         for( TNatural m = 0; m < M; ++m )
         {
         std::cout << (m+1) << "/" << M << std::endl;

         TColumnMap x( bI, N, 1 );
         auto rIt = this->m_Relation.upper_bound( x );
         bool ok = ( rIt == this->m_Relation.end( ) );
         if( !ok )
         ok = ( this->m_ConvergenceThreshold < std::sqrt( ( rIt->first - x ).array( ).pow( 2 ).sum( ) ) );
         if( ok )
         {
         this->_mean( bM, bI, I, N, M );
         this->m_Relation.insert( std::make_pair( x, TColumnMap( bM, N, 1 ) ) );
         } // end if

         bI = bI + N;
         bM = bM + N;
         } // end for
      */

      /* TODO
         TMatrixMap D( I, N, M );

         // Prepare outputs
         this->m_Means.clear( );
         this->m_Relation.clear( );

         // Main loop
         for( TNatural c = 0; c < D.cols( ); ++c )
         {
         TColumnMap x( D.col( c ).data( ), D.rows( ), 1 );
         auto rIt = this->m_Relation.upper_bound( x );
         bool ok = ( rIt == this->m_Relation.end( ) );
         if( !ok )
         ok = ( this->m_ConvergenceThreshold < std::sqrt( ( rIt->first - x ).array( ).pow( 2 ).sum( ) ) );
         if( ok )
         {
         for( TNatural n = 0; n < N; ++n )
         this->m_Means.push_back( 0 );
         TColumnMap s( this->m_Means.data( ) + ( this->m_Means.size( ) - N ), N, 1 );
         this->_mean( s, x, D );
         this->m_Relation.insert( std::make_pair( x, s ) );

         std::cout << x.transpose( ) << " ++++++++++++ " << s.transpose( ) << std::endl;

         } // end if
         } // end for
         this->m_Means.shrink_to_fit( );

         std::cout << "----------------------------" << std::endl;
         for( auto m: this->m_Relation )
         std::cout << m.first.transpose( ) << " ***** " << m.second.transpose( ) << std::endl;
         std::cout << "----------------------------" << std::endl;
         for( auto v: this->m_Means )
         std::cout << v << std::endl;
         std::cout << "----------------------------" << std::endl;
      */
    }

  void shift( TReal* bO, const TReal* bI, const TNatural& N, const TNatural& M )
    {
      /* TODO
         const TReal* I = bI;
         TReal* O = bO;
         for( TNatural m = 0; m < M; ++m )
         {
         TColumnMap x( I, N, 1 );
         auto rIt = this->m_Relation.upper_bound( x );
         if( rIt == this->m_Relation.end( ) )
         rIt = this->m_Relation.begin( );

         std::cout << rIt->first.transpose( ) << " ::: " << rIt->second.transpose( ) << std::endl;

         for( TNatural n = 0; n < N; ++n )
         O[ n ] = rIt->second( n, 0 );

         I += N;
         O += N;
         } // end for
      */
    }


protected:

  void _init( )
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

  void _mean( TReal* s, const TReal* x, const TReal* I, const TNatural& N, const TNatural& M )
    {
      // **NOTE** s and cur point to the same data buffer
      /* TODO
         Eigen::Map< TColumn > cur( const_cast< TReal* >( s.data( ) ), s.rows( ), s.cols( ) );
         TColumn pre;
         cur = x;

         // Main loop
         bool stop = false;
         TNatural i = 0;
         while( !stop && i <= this->m_MaximumNumberOfIterations )
         {
         i++;
         pre = cur;

         TReal W = TReal( 0 );
         TColumn mean = TColumn::Zero( x.rows( ), x.cols( ) );

         for( TNatural c = 0; c < D.cols( ); ++c )
         {
         TReal w = this->m_Kernel( this->m_Distance( TColumnMap( D.col( c ).data( ), D.rows( ), 1 ), s ) );
         if( w > this->m_Epsilon )
         {
         mean += D.col( c ) * w;
         W += w;
         } // end if
         } // end for
         if( W != TReal( 0 ) )
         {
         cur = mean / W;

         if( std::sqrt( ( cur - pre ).array( ).pow( 2 ).sum( ) ) < this->m_ConvergenceThreshold )
         stop = true;
         }
         else
         stop = true;
         } // end while
      */
    }

protected:
  TDistance m_Distance;
  TKernel   m_Kernel;

  TReal    m_Epsilon;
  TReal    m_ConvergenceThreshold;
  TNatural m_MaximumNumberOfIterations { 100 };
  TReal    m_ClusterMergeThreshold     { TReal( 1 ) };

  TMatrixMap m_Data { nullptr, 0, 0 };

  /* TODO
     TReal*          m_Means { nullptr };
     TColumnRelation m_Relation;
  */
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

  MeanShiftFunctions< TReal, TNatural > ms_funcs( I, N, M );
  ms_funcs.compute_means( );

  /* TODO
     std::vector< TReal > tI( N * M, 0 );
     ms_funcs.shift( tI.data( ), I, N, M );
     for( auto v: tI )
     std::cout << v << std::endl;
  */


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
