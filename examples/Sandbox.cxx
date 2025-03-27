// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <ivqML/IO.h>
#include <boost/any.hpp>
#include <vector>
#include <boost/type_erasure/any.hpp>
#include <boost/type_erasure/operators.hpp>


class Layer
{
public:
  using TReal = double;
  using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
  using TColumn = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;

  Layer( int rows, int cols )
    {
      W = TMatrix::Zero( rows, cols );
      B = TColumn::Zero( rows );
    }

  template< class _TX >
  auto operator()( const Eigen::EigenBase< _TX >& X ) const
    {
      return( ( W * X.derived( ).template cast< TReal >( ) ).colwise( ) + B );

      // A.push_back( t );
      /* TODO
         for( unsigned int l = 1; l <= W.size( ); ++l )
         {
         Z.push_back( boost::any( ( W[ l ] * A.back( ) ).colwise( ) + B[ l ] ) );
         A.push_back( Z.back( ) );
         } // end for
      */
      /* TODO
         auto A = ( Eigen::Map< TMatrix >( W[ l ].data( ), W[ l ].rows( ), W[ l ].cols( ) ) * X.derived( ) ).colwise( ) + Eigen::Map< TColumn >( B[ l ].data( ), B[ 0 ].rows( ), B[ l ].cols( ) );
         // if( l < W.size( ) - 1 )
         return( this->operator()( A, l + 1 ) );
         // else
         // return( A );
         llsls
      */
      // return( t ); // A.back( ) );
    }

protected:
  mutable TMatrix W;
  mutable TColumn B;
};

int main( int argc, char** argv )
{
  using TReal = double;
  using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
  using TColumn = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;

  TMatrix Xtr, Ytr, Xte, Yte;

  ivqML::IO::ReadMNIST( Xtr, Ytr, Xte, Yte, argv[ 1 ] );
  
  TMatrix W = TMatrix::Zero( 40, Xtr.rows( ) );
  TColumn B = TColumn::Zero( 40 );

  Layer l1( 40, Xtr.rows( ) );
  Layer l2( 20, 40 );
  Layer l3( 10, 20 );
  auto Y = l3( l2( l1( Xtr ) ) );
  std::cout << "_Z" << typeid( Y ).name( ) << std::endl;
  std::cout << Y.rows( ) << " " << Y.cols( ) << std::endl;

  /* TODO
     auto Y = ( Eigen::Map< TMatrix >( W.data( ), W.rows( ), W.cols( ) ) * Xtr ).colwise( ) + Eigen::Map< TColumn >( B.data( ), B.rows( ), B.cols( ) );

     std::cout << Xtr.rows( ) << " " << Xtr.cols( ) << std::endl;
     std::cout << Y.rows( ) << " " << Y.cols( ) << std::endl;
     std::cout << "_Z" << typeid( Y ).name( ) << std::endl;
     Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<TReal, TReal>, const Eigen::Product<TMatrixMap, TMatrix, 0>, const Eigen::Replicate<TColumnMap, 1, -1> >
  */

  /* TODO
     std::cout << Xtr.rows( ) << " " << Xtr.cols( ) << std::endl;
     std::cout << Ytr.rows( ) << " " << Ytr.cols( ) << std::endl;
     std::cout << Xte.rows( ) << " " << Xte.cols( ) << std::endl;
     std::cout << Yte.rows( ) << " " << Yte.cols( ) << std::endl;
  */

  /* TODO
     std::cout << "P2" << std::endl;
     std::cout << "28 28" << std::endl;
     std::cout << "255" << std::endl;
     for( unsigned long long c = 0; c < Xtr.cols( ); ++c )
     std::cout << int( Xtr( 55554, c ) ) << " ";
     std::cout << std::endl;
  */


  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
