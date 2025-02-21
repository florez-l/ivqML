// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Helpers__ConfussionMatrix__h__
#define __ivqML__Helpers__ConfussionMatrix__h__

#include <ivqML/Config.h>
#include <tuple>

namespace ivqML
{
  namespace Helpers
  {
    /**
     */
    template< class _Tx, class _Ty, class _TR = long double >
    auto BinaryConfussionMatrix(
      const Eigen::EigenBase< _Tx >& bx, const Eigen::EigenBase< _Ty >& by )
    {
      using _M = Eigen::Matrix< _TR, Eigen::Dynamic, Eigen::Dynamic >;

      auto x = bx.derived( ).template cast< _TR >( );
      auto y = by.derived( ).template cast< _TR >( );
      auto cx = ( _TR( 1 ) - x.array( ) ).matrix( );
      auto cy = ( _TR( 1 ) - y.array( ) ).matrix( );

      _TR TP = ( cx.transpose( ) * cy.matrix( ) )( 0 , 0 );
      _TR TN = ( x.transpose( ) * y.matrix( ) )( 0 , 0 );
      _TR FP = ( cx.transpose( ) * y.matrix( ) )( 0 , 0 );
      _TR FN = ( x.transpose( ) * cy.matrix( ) )( 0 , 0 );

      _TR sen = 0, spe = 0, acc = 0, f1s = 0;
      if( ( TP + FN ) != 0 )
        sen = TP / ( TP + FN );
      if( ( TN + FP ) != 0 )
        spe = TN / ( TN + FP );
      if( ( TP + FP ) != 0 )
        acc = TP / ( TP + FP );
      if( ( TP + ( ( FP + FN ) / 2 ) ) != 0 )
        f1s = TP / ( TP + ( ( FP + FN ) / 2 ) );

      _M K( 2, 2 );
      K( 0 , 0 ) = TP;
      K( 1 , 1 ) = TN;
      K( 1 , 0 ) = FP;
      K( 0 , 1 ) = FN;

      return( std::make_tuple( K, sen, spe, acc, f1s ) );
    }
  } // end namespace
} // end namespace

#endif // __ivqML__Helpers__ConfussionMatrix__h__

// eof - $RCSfile$
