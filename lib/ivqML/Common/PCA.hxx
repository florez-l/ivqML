// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Common__PCA__hxx__
#define __ivqML__Common__PCA__hxx__

#include <iterator>
#include <numeric>
#include <Eigen/SVD>

// -------------------------------------------------------------------------
template< class _TData, class _TReal >
auto ivqML::Common::
EigenAnalysis( const Eigen::EigenBase< _TData >& X )
{
  auto D = X.derived( ).template cast< _TReal >( );
  auto m = D.colwise( ).mean( ).eval( );
  auto C = D.rowwise( ) - m;
  auto S = ( ( C.transpose( ) * C ) / _TReal( D.rows( ) - 1 ) ).eval( );

  Eigen::BDCSVD< decltype( S ) > E;
  E.compute( S, Eigen::ComputeFullV );

  return( std::make_tuple( m, E.matrixV( ), E.singularValues( ) ) );
}

// -------------------------------------------------------------------------
template< class _TData, class _TMean, class _TMatrix, class _TValues, class _TReal >
auto ivqML::Common::
PCA(
  const Eigen::EigenBase< _TData >& X,
  const std::tuple< _TMean, _TMatrix, _TValues >& E,
  const _TReal& p
  )
{
  auto V = ( std::get< 2 >( E ).array( ) / std::get< 2 >( E ).sum( ) ).eval( );
  std::partial_sum( V.data( ), V.data( ) + V.size( ), V.data( ) );

  unsigned int l = std::distance(
    V.data( ),
    std::find_if_not(
      V.data( ), V.data( ) + V.size( ),
      [=]( const _TReal& v )
      {
        return( v < p );
      }
      )
    );

  std::vector< Eigen::Index > idx( l + 1 );
  std::iota( idx.begin( ), idx.end( ), 0 );

  return(
    std::make_pair(
      V( l ),
      (
        (
          X.derived( ).template cast< _TReal >( ).rowwise( )
          -
          std::get< 0 >( E )
          )
        *
        std::get< 1 >( E )
        )
      ( ivq_EIGEN_ALL, idx ).eval( )
      )
    );
}

// -------------------------------------------------------------------------
template< class _TData, class _TReal >
auto ivqML::Common::
PCA( const Eigen::EigenBase< _TData >& X, const _TReal& p )
{
  return(
    ivqML::Common::PCA(
      X, ivqML::Common::EigenAnalysis< _TData, _TReal >( X ), p
      )
    );
}

#endif // __ivqML__Common__PCA__hxx__

// eof - $RCSfile$
