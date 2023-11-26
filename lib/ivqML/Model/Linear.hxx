// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Linear__hxx__
#define __ivqML__Model__Linear__hxx__

#include <Eigen/Dense>

// -------------------------------------------------------------------------
/* TODO
   template< class _S >
   template< class _G, class _X, class _Y >
   void ivqML::Model::Linear< _S >::
   cost(
   Eigen::EigenBase< _G >& iG,
   const Eigen::EigenBase< _X >& iX,
   const Eigen::EigenBase< _Y >& iY,
   TScalar* J
   ) const
   {
   using _Gs = typename _G::Scalar;

   auto X = iX.derived( ).template cast< TScalar >( );
   auto Y = iY.derived( ).template cast< TScalar >( );
   TMatrix D = this->evaluate( X ) - Y.array( );

   iG.derived( )( 0, 0 ) = ( _Gs )( TScalar( 2 ) * D.mean( ) );
   iG.derived( ).block( 0, 1, 1, iG.cols( ) - 1 )
   =
   ( ( D.transpose( ) * X ) * ( TScalar( 2 ) / TScalar( X.rows( ) ) ) )
   .template cast< _Gs >( );
   if( J != nullptr )
   *J = D.array( ).pow( 2 ).mean( );
   }
*/

// -------------------------------------------------------------------------
template< class _S >
template< class _Y, class _X >
void ivqML::Model::Linear< _S >::
fit(
  const Eigen::EigenBase< _X >& iX, const Eigen::EigenBase< _Y >& iY,
  const TScalar& l
  )
{
  /* TODO
     auto X = iX.derived( ).template cast< TScalar >( );
     auto Y = iY.derived( ).template cast< TScalar >( );

     TScalar m = TScalar( X.rows( ) );
     TNatural n = TScalar( X.cols( ) );
     this->set_number_of_inputs( n );

     TMatrix Xi( X.rows( ), X.cols( ) + 1 );
     Xi << TMatrix::Ones( X.rows( ), 1 ), X;

     TMap( this->m_T.get( ), 1, n + 1 ) =
     (
     Y.transpose( ) * Xi
     *
     (
     (
     ( Xi.transpose( ) * Xi ) / m )
     +
     ( TMatrix::Identity( n + 1, n + 1 ) * l )
     ).inverse( )
     ) / m;
  */
}

#endif // __ivqML__Model__Linear__hxx__

// eof - $RCSfile$
