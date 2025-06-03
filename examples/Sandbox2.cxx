// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <algorithm>
#include <map>
#include <ivqML/Config.h>

#include <itkVectorImage.h>
#include <ivq/ITK/EigenUtils.h>
#include <ivq/ITK/ImageFileReader.h>

static const unsigned int VDim = 2;
using TNatural = unsigned long long;
using TReal = double;
using TImage = itk::VectorImage< TReal, VDim >;
using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
using TCol = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;

// -------------------------------------------------------------------------
TCol performMeanShift( const TCol& c, const TMatrix& I, double bandwidth, double convergence_threshold, int max_iterations )
{
  TCol current_point = c; // Start the shifting process from the initial point
  TCol prev_point;                    // To store the point from the previous iteration for convergence check

  for (int iter = 0; iter < max_iterations; ++iter)
  {
    prev_point = current_point; // Save the current point before shifting

    auto W =
      ( I.colwise( ) - current_point ).array( ).pow( 2 ).colwise( ).sum( ).array( ).sqrt( ).unaryExpr(
        [ &bandwidth ]( const TReal& d ) -> TReal
        {
          TReal w = 0;
          if( bandwidth > 1e-6)
          {
            w = std::exp(-0.5 * std::pow(d / bandwidth, 2));
            if( w <= 1e-6 )
              w = TReal( 0 );
          }
          return( w );
        }
        );
    TReal tW = W.sum( );
    if( tW <= 1e-6 )
      break;
    current_point = ( I.array( ).rowwise( ) * W.array( ).row( 0 ) ).matrix( ).rowwise( ).sum( ) / tW;

    // Check for convergence: if the shift is smaller than the threshold, stop iterating
    if( std::sqrt( ( current_point - prev_point ).array( ).pow( 2 ).sum( ) ) < convergence_threshold)
    {
      break;
    }
  }
  return( current_point );
}

// -------------------------------------------------------------------------
template< class _TX >
void meanShiftClustering(
  const Eigen::EigenBase< _TX >& bI, double bandwidth, double convergence_threshold, int max_iterations, double cluster_merge_threshold
  )
{
  auto I = bI.derived( ).template cast< TReal >( );
  TMatrix shifted_modes = TMatrix::Zero( I.rows( ), I.cols( ) );

  // Step 1: Perform Mean Shift for each data point
  // Each original data point starts its own "hill-climbing" process
  for( unsigned long long c = 0; c < I.cols( ); ++c )
  {
    // std::cout << ( c + 1 ) << "/" << I.cols( ) << std::endl;
    shifted_modes.col( c ) = performMeanShift( I.col( c ), I, bandwidth, convergence_threshold, max_iterations );
    // std::cout << "\t" << I.col( c ).transpose( ) << " <-> " << shifted_modes.col( c ).transpose( ) << std::endl;
  }

  // Step 2: Group the converged modes into distinct clusters
  // If multiple initial points converge to very similar modes, they belong to the same cluster.
  unsigned long long R = shifted_modes.rows( );
  std::vector< TReal > cluster_centers_data( shifted_modes.data( ), shifted_modes.data( ) + R );
  Eigen::Map< TMatrix > cluster_centers( cluster_centers_data.data( ), R, cluster_centers_data.size( ) / R );
  int current_cluster_id = 0;
  for( size_t i = 1; i < shifted_modes.cols( ); ++i )
  {
    Eigen::Index minI;
    TReal minD = ( cluster_centers.colwise( ) - shifted_modes.col( i ) ).array( ).pow( 2 ).colwise( ).sum( ).array( ).sqrt( ).minCoeff( &minI );
    if( minD >= cluster_merge_threshold )
    {
      cluster_centers_data.insert( cluster_centers_data.end( ), shifted_modes.data( ) + ( i * R ), shifted_modes.data( ) + ( ( i * R ) + R ) );
      new ( &cluster_centers ) Eigen::Map< TMatrix >( cluster_centers_data.data( ), R, cluster_centers_data.size( ) / R );
    } // end if
  }
  
  std::cout << cluster_centers_data.size( ) << std::endl;
  std::cout << cluster_centers.rows( ) << " "  << cluster_centers.cols( ) << std::endl;
  std::cout << I.rows( ) << " "  << I.cols( ) << std::endl;
  std::cout << "-------------------------------------------" << std::endl;
  std::cout << cluster_centers << std::endl;

  /* TODO
  // Optional: Recalculate true centroids of clusters based on assigned original points
  // This can provide more accurate cluster centers than just the converged modes.
  std::vector<Point> final_cluster_centroids(current_cluster_id);
  std::vector<int> cluster_point_counts(current_cluster_id, 0);

  for (const auto& point : data) {
  if (point.cluster_id != -1) { // Only consider assigned points
  if (final_cluster_centroids[point.cluster_id].dimension == 0) {
  // Initialize if first point for this cluster
  final_cluster_centroids[point.cluster_id] = Point(std::vector<double>(point.dimension, 0.0));
  final_cluster_centroids[point.cluster_id].dimension = point.dimension;
  final_cluster_centroids[point.cluster_id].cluster_id = point.cluster_id;
  }
  for (int d = 0; d < point.dimension; ++d) {
  final_cluster_centroids[point.cluster_id].coords[d] += point.coords[d];
  }
  cluster_point_counts[point.cluster_id]++;
  }
  }

  for (int i = 0; i < current_cluster_id; ++i) {
  if (cluster_point_counts[i] > 0) {
  for (int d = 0; d < final_cluster_centroids[i].dimension; ++d) {
  final_cluster_centroids[i].coords[d] /= cluster_point_counts[i];
  }
  }
  }

  return final_cluster_centroids; // Return the final, more accurate cluster centroids
  */



}
// std::back_inserter( shifted_means ), I, m, N, M, distance, kernel )

void SingleSampleMeanShift( std::back_insert_iterator< std::vector< TReal > > shifted_mean, const TReal* I, TNatural m, TNatural N, TNatural M, std::function< TReal( const TReal*, const TReal*, const TNatural& ) > distance, std::function< TReal( const TReal& ) > kernel )
{
  TCol current_point = Eigen::Map< const TCol >( I + ( m * N ), N, 1 );
  TCol prev_point;

  for( TNatural iter = 0; iter < 100 /*max_iterations*/; ++iter)
  {
    prev_point = current_point; // Save the current point before shifting

    TReal W = TReal( 0 );
    TCol mean = TCol::Zero( N, 1 );
    for( TNatural i = 0; i < M; ++i )
    {
      TReal w = kernel( distance( I + ( i * N ), current_point.data( ), N ) );
      if( w > TReal( 0 ) )
      {
        W += w;
        mean += Eigen::Map< const TCol >( I + ( i * N ), N, 1 );
      } // end if
    } // end for
    std::cout << "\t" << mean.transpose( ) << ":" << W << std::endl;
    if( W > TReal( 0 ) )
      current_point = mean / W;
    else
      break;
    if( std::sqrt( ( current_point - prev_point ).array( ).pow( 2 ).sum( ) ) < 1e-3 /*convergence_threshold*/)
      break;
  } // end for

  std::cout << current_point.transpose( ) << std::endl;
  for( TNatural d = 0; d < current_point.size( ); ++d )
    *shifted_mean = current_point( d );

  std::exit( 1 );
}

void MeanShift( const TReal* I, TNatural N, TNatural M, std::function< TReal( const TReal*, const TReal*, const TNatural& ) > distance, std::function< TReal( const TReal& ) > kernel )
{
  using TColMap = Eigen::Map< const TCol >;
  struct SCmp
  {
    bool operator()( const TColMap& a, const TColMap& b ) const
      {
        return( std::lexicographical_compare( a.data( ), a.data( ) + a.size( ), b.data( ), b.data( ) + b.size( ) ) );
      }
  };
  std::map< TColMap, TNatural, SCmp > shifted_means_relations;
  std::vector< TReal > shifted_means;

  for( TNatural m = 0; m < M; ++m )
  {
    TColMap x( I + ( m * N ), N, 1 );
    if( shifted_means_relations.find( x ) == shifted_means_relations.end( ) )
    {
      std::cout << ( m + 1 ) << " / " << M << std::endl;
      SingleSampleMeanShift( std::back_inserter( shifted_means ), I, m, N, M, distance, kernel );
      shifted_means_relations.insert( std::make_pair( x, shifted_means.size( ) / N ) );
    } // end if
  } // end for

  for( auto v: shifted_means )
    std::cout << v << " ";
  std::cout << std::endl;
  
}


int main( int argc, char** argv )
{
  std::cout << "Number of threads: " << Eigen::nbThreads( ) << std::endl;

  /* TODO
     TImage::Pointer input;
     {
     auto r = ivq::ITK::ImageFileReader< TImage >::New( );
     r->SetFileName( argv[ 1 ] );
     r->Update( );
     input = r->GetOutput( );
     input->DisconnectPipeline( );
     }
     auto I = ivq::ITK::ImageToMatrix( input.GetPointer( ) );
  */
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

  // Parameters for the Mean Shift algorithm
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
