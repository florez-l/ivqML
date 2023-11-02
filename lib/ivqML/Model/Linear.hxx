// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Linear__hxx__
#define __ivqML__Model__Linear__hxx__

#include <Eigen/Dense>

// -------------------------------------------------------------------------
template< class _S >
template< class _Y, class _X >
void ivqML::Model::Linear< _S >::
operator()(
  Eigen::EigenBase< _Y >& iY, const Eigen::EigenBase< _X >& iX,
  bool derivative
  ) const
{
  using _YS = typename _Y::Scalar;
  using _YM = Eigen::Matrix< _YS, Eigen::Dynamic, Eigen::Dynamic >;

  auto X = iX.derived( ).template cast< _S >( );
  if( derivative )
    iY.derived( ) << _YM::Ones( X.rows( ), 1 ), X.template cast< _YS >( );
  else
    iY.derived( ) =
      ( ( X * this->m_cT ).array( ) + this->m_T.get( )[ 0 ] )
      .template cast< _YS >( );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _Y, class _X >
void ivqML::Model::Linear< _S >::
fit(
  const Eigen::EigenBase< _X >& iX, const Eigen::EigenBase< _Y >& iY,
  const _S& l
  )
{
  auto X = iX.derived( ).template cast< _S >( );
  auto Y = iY.derived( ).template cast< _S >( );

  TNatural m = X.rows( );
  TNatural n = X.cols( );
  this->set_number_of_inputs( n );

  TMatrix R = TMatrix::Zero( this->m_P, this->m_P );
  R( 0, 0 ) = 1;
  R.block( 1, 1, n, n ) = ( X.transpose( ) * X ).array( ) / _S( m );
  R.block( 0, 1, 1, n ) = X.colwise( ).mean( );
  R.block( 1, 0, n, 1 ) = R.block( 0, 1, 1, n ).transpose( );

  if( l != _S( 0 ) )
  {
    /* TODO
       L = numpy.identity( n + 1 ) * l
       L[ 0 , 0 ] = 0
       R += L
    */
  } // end if

  TMatrix c = TMatrix::Zero( 1, this->m_P );
  c( 0, 0 ) = Y.mean( );
  c.block( 0, 1, 1, n ) =
    ( X.array( ).colwise( ) * Y.array( ).col( 0 ) ).colwise( ).mean( );
  this->m_nT = c * R.inverse( );
}

#endif // __ivqML__Model__Linear__hxx__

// eof - $RCSfile$
