// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <iostream>
#include <random>

#include <itkImage.h>
#include <ivq/ITK/EigenUtils.h>
#include <ivq/ITK/ImageFileReader.h>

#include <ivqML/Model/Logistic.h>
#include <ivqML/Cost/CrossEntropy.h>
#include <ivqML/Optimizer/GradientDescent.h>

using _R = long double;
using _M = ivqML::Model::Logistic< _R >;
using _I = itk::Image< _R, 2 >;

int main( int argc, char** argv )
{
  unsigned int m = 50;

  // Get input data
  auto reader = ivq::ITK::ImageFileReader< _I >::New( );
  reader->SetFileName( argv[ 1 ] );
  reader->NormalizeOn( );
  reader->Update( );
  auto Iy = ivq::ITK::ImageToMatrix( reader->GetOutput( ) ).transpose( );
  auto roi = reader->GetOutput( )->GetRequestedRegion( );
  unsigned long long Iw = roi.GetSize( )[ 0 ];
  unsigned long long Ih = roi.GetSize( )[ 1 ];
  unsigned long long Is = Iw * Ih;

  _M::TMatrix rows( Ih, 1 ), cols( Iw, 1 );
  rows.col( 0 ).setLinSpaced( Ih, 0, Ih - 1 );
  cols.col( 0 ).setLinSpaced( Iw, 0, Iw - 1 );
  _M::TMatrix Ix( Is, 2 );
  Ix
    <<
    cols.replicate( 1, Ih ).reshaped( Is, 1 ),
    rows.replicate( 1, Iw ).transpose( ).reshaped( Is, 1 );

  struct IyVisitor
  {
    std::vector< Eigen::Index > zeros, ones;
    void init( const _R& y, const Eigen::Index& i, const Eigen::Index& j )
      {
        zeros.clear( );
        ones.clear( );
        this->operator()( y, i, j );
      }
    void operator()( const _R& y, const Eigen::Index& i, const Eigen::Index& j )
      {
        if( y < 0.5 ) zeros.push_back( i );
        else          ones.push_back( i );
      }
  } Iy_visitor;
  Iy.visit( Iy_visitor );

  // Shuffle indices and get 'm' samples
  std::random_device dev;
  std::mt19937 eng( dev( ) );
  std::shuffle( Iy_visitor.zeros.begin( ), Iy_visitor.zeros.end( ), eng );
  std::shuffle( Iy_visitor.ones.begin( ), Iy_visitor.ones.end( ), eng );
  Iy_visitor.zeros.erase( Iy_visitor.zeros.begin( ) + m, Iy_visitor.zeros.end( ) );
  Iy_visitor.ones.erase( Iy_visitor.ones.begin( ) + m, Iy_visitor.ones.end( ) );

  _M::TMatrix X( m << 1, Ix.cols( ) );
  _M::TMatrix Y( m << 1, Iy.cols( ) );
  X << Ix( Iy_visitor.zeros, ivq_EIGEN_ALL ), Ix( Iy_visitor.ones, ivq_EIGEN_ALL );
  Y << _M::TMatrix::Zero( m, 1 ), _M::TMatrix::Ones( m, 1 );

  // Model to be fitted
  _M fitted_model( 2 );
  fitted_model.random_fill( );
  std::cout << "Initial model : " << fitted_model << std::endl;

  // Optimization algorithm
  using _C = ivqML::Cost::CrossEntropy< _M >;
  ivqML::Optimizer::GradientDescent< _C > opt( fitted_model, X, Y );
  opt.set_debug(
    []( const _R& J, const _R& G, const _M* m, const _M::TNatural& i )
    -> bool
    {
      std::cout << "J=" << J << ", Gn=" << G << ", i=" << i << std::endl;
      return( false );
    }
    );
  opt.fit( );
  std::cout << "Fitted model  : " << fitted_model << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
