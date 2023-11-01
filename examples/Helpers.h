// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__examples__Helpers__h__
#define __ivqML__examples__Helpers__h__

#include <ivq/ITK/EigenUtils.h>

/* TODO
   #include <algorithm>
   #include <iostream>
   #include <random>

   #include <itkImage.h>
   #include <ivq/ITK/ImageFileReader.h>

   #include <ivqML/Model/Logistic.h>
   #include <ivqML/Cost/CrossEntropy.h>
   #include <ivqML/Optimizer/ADAM.h>

   using _R = long double;
   using _M = ivqML::Model::Logistic< _R >;
   using _I = itk::Image< _R, 2 >;
*/
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
      auto Iy = ivq::ITK::ImageToMatrix( image ); // .template cast< _S >( );
      auto roi = image->GetRequestedRegion( );
      unsigned long long Iw = roi.GetSize( )[ 0 ];
      unsigned long long Ih = roi.GetSize( )[ 1 ];
      unsigned long long Is = Iw * Ih;

      // Image's tessellation
      // TODO: this is **almost** dimension-generic -> work more on linSpaced!
      _M rows( Ih, 1 ), cols( Iw, 1 );
      rows.col( 0 ).setLinSpaced( Ih, 0, Ih - 1 );
      cols.col( 0 ).setLinSpaced( Iw, 0, Iw - 1 );
      _M Ix( Is, _I::ImageDimension );
      Ix
        <<
        cols.replicate( 1, Ih ).reshaped( Is, 1 ),
        rows.replicate( 1, Iw ).transpose( ).reshaped( Is, 1 );


      std::cout << ( Iy.array( ) == 0 ) << std::endl;

      /* TODO
         struct S_Iy_visitor
         {
         std::vector< std::vector< Eigen::Index > > labels;

         S_Iy_visitor( unsigned int n_labels )
         {
         labels.clear( );
         labels.resize( n_labels );
         }
         void init( const _S& y, const Eigen::Index& i, const Eigen::Index& j )
         {
         this->operator()( y, i, j );
         }
         void operator()(
         const _S& y, const Eigen::Index& i, const Eigen::Index& j
         )
         {
         if( y < 0.5 ) zeros.push_back( i );
         else          ones.push_back( i );
         }
         } Iy_visitor( n_labels );
         Ix.visit( Iy_visitor );
      */
      



      _M D( n_samples * n_labels, _I::ImageDimension + Iy.rows( ) );
      return( D );

      /* TODO


         // Shuffle indices and get 'm' samples
         std::random_device dev;
         std::mt19937 eng( dev( ) );
         std::shuffle( Iy_visitor.zeros.begin( ), Iy_visitor.zeros.end( ), eng );
         std::shuffle( Iy_visitor.ones.begin( ), Iy_visitor.ones.end( ), eng );
         Iy_visitor.zeros.erase(
         Iy_visitor.zeros.begin( ) + m, Iy_visitor.zeros.end( )
         );
         Iy_visitor.ones.erase(
         Iy_visitor.ones.begin( ) + m, Iy_visitor.ones.end( )
         );

         _M::TMatrix X( m << 1, Ix.cols( ) );
         _M::TMatrix Y( m << 1, Iy.cols( ) );
         X <<
         Ix( Iy_visitor.zeros, ivq_EIGEN_ALL ),
         Ix( Iy_visitor.ones, ivq_EIGEN_ALL );
         Y << _M::TMatrix::Zero( m, 1 ), _M::TMatrix::Ones( m, 1 );

         // Model to be fitted
         _M fitted_model( 2 );
         fitted_model.random_fill( );
         std::cout << "Initial model : " << fitted_model << std::endl;

         // Optimization algorithm
         using _C = ivqML::Cost::CrossEntropy< _M >;
         ivqML::Optimizer::ADAM< _C > opt( fitted_model, X, Y );
         opt.set_parameter( "alpha", 1e-3 );
         opt.set_parameter( "debug_iterations", 100000 );
         opt.set_debug(
         []( const _R& J, const _R& G, const _M* m, const _M::TNatural& i )
         -> bool
         {
         std::cerr << "J=" << J << ", Gn=" << G << ", i=" << i << std::endl;
         return( false );
         }
         );
         opt.fit( );
         std::cout << "Fitted model  : " << fitted_model << std::endl;

         return( EXIT_SUCCESS );
      */
    }
  } // end namespace
} // end namespace

#endif // __ivqML__examples__Helpers__h__

// eof - $RCSfile$
