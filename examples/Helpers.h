// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__examples__Helpers__h__
#define __ivqML__examples__Helpers__h__

#include <algorithm>
#include <numeric>
#include <random>
#include <ivqML/Common/ImageHelpers.h>

namespace ivqML
{
  namespace Helpers
  {
    // ---------------------------------------------------------------------
    template< class _I, class _M >
    _M extract_discrete_samples_from_image(
      const _I* image, unsigned int n_samples, unsigned int n_labels
      )
    {
      using _S = typename _M::Scalar;

      // Data from image
      _M Ix = ivqML::Common::ImageHelpers::meshgrid< _I, _M >( image );

      auto Iy =
        ivq::ITK::ImageToMatrix( image ).row( 0 ).template cast< _S >( );
      _S mY = Iy.minCoeff( );
      _S cY = _S( n_labels - 1 ) / ( Iy.maxCoeff( ) - mY );
      auto Ly =
        ( ( Iy.array( ) - mY ) * cY ).template cast< unsigned char >( );
      std::vector< Eigen::Index > idx( Ly.cols( ) );
      idx.shrink_to_fit( );
      std::iota( idx.begin( ), idx.end( ), 0 );
      std::sort(
        idx.begin( ), idx.end( ),
        [&]( const Eigen::Index& a, const Eigen::Index& b ) -> bool
        {
          return( Ly( 0, a ) < Ly( 0, b ) );
        }
        );

      std::random_device dev;
      std::mt19937 eng( dev( ) );
      std::vector< Eigen::Index > labels;
      unsigned long long i = 0;
      for( unsigned long long l = 0; l < n_labels; ++l )
      {
        unsigned long long s =
          ( Ly.array( ) == l ).template cast< unsigned long long >( ).sum( );
        std::shuffle( idx.begin( ) + i, idx.begin( ) + ( i + s ), eng );
        labels.insert(
          labels.end( ), idx.begin( ) + i, idx.begin( ) + ( i + n_samples )
          );
        i = s;
      } // end if
      labels.shrink_to_fit( );
      std::shuffle( labels.begin( ), labels.end( ), eng );

      _M D( n_samples * n_labels, _I::ImageDimension + Iy.rows( ) );
      D <<
        Ix( labels, ivq_EIGEN_ALL ),
        Ly( 0, labels ).transpose( ).template cast< _S >( );
      return( D );
    }
  } // end namespace
} // end namespace

#endif // __ivqML__examples__Helpers__h__

// eof - $RCSfile$
