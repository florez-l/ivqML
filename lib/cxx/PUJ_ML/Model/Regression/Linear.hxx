// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Regression__Linear__hxx__
#define __PUJ_ML__Model__Regression__Linear__hxx__

#include <Eigen/Dense>

// -------------------------------------------------------------------------
template< class _R >
PUJ_ML::Model::Regression::Linear< _R >::
Linear( const unsigned long long& n )
  : Superclass( n )
{
}

// -------------------------------------------------------------------------
template< class _R >
unsigned long long PUJ_ML::Model::Regression::Linear< _R >::
number_of_inputs( ) const
{
  return( this->m_P.size( ) - 1 );
}

// -------------------------------------------------------------------------
template< class _R >
void PUJ_ML::Model::Regression::Linear< _R >::
init( const unsigned long long& n )
{
  this->Superclass::init( n + 1 );
}

// -------------------------------------------------------------------------
template< class _R >
template< class _Y, class _X >
void PUJ_ML::Model::Regression::Linear< _R >::
evaluate( Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X ) const
{
  Y.derived( ) =
    (
      (
        X.derived( ).template cast< _R >( )
        *
        ConstMCol( this->m_P.data( ) + 1, this->m_P.size( ) - 1, 1 )
        ).array( ) + this->m_P[ 0 ]
      )
    .template cast< typename _Y::Scalar >( );
}

// -------------------------------------------------------------------------
template< class _R >
template< class _X, class _Y >
void PUJ_ML::Model::Regression::Linear< _R >::
fit( const Eigen::EigenBase< _X >& X, const Eigen::EigenBase< _Y >& Y )
{
  unsigned long long n = X.cols( );
  unsigned long long m = X.rows( );
  this->init( n );

  auto iX = X.derived( ).template cast< _R >( );
  auto iY = Y.derived( ).template cast< _R >( );

  TMatrix A( n + 1, n + 1 );
  A( 0, 0 ) = 1;
  A.block( 0, 1, 1, n ) = iX.colwise( ).mean( );
  A.block( 1, 0, n, 1 ) = A.block( 0, 1, 1, n ).transpose( );
  A.block( 1, 1, n, n ) = ( iX.transpose( ) * iX ) / TReal( m );

  TCol B( n + 1 );
  B( 0 ) = iY.mean( );
  B.block( 1, 0, n, 1 ) =
    ( iX.array( ).colwise( ) * iY.array( ).col( 0 ) ).
    colwise( ).mean( ).transpose( );

  MCol( this->m_P.data( ), n + 1, 1 ) = A.inverse( ) * B;
}

// -------------------------------------------------------------------------
template< class _R >
PUJ_ML::Model::Regression::Linear< _R >::Cost::
Cost( TModel* m )
  : m_Model( m )
{
}

// -------------------------------------------------------------------------
template< class _R >
template< class _X, class _Y >
typename PUJ_ML::Model::Regression::Linear< _R >::
TReal PUJ_ML::Model::Regression::Linear< _R >::Cost::
evaluate(
  const Eigen::EigenBase< _X >& X,
  const Eigen::EigenBase< _Y >& Y,
  TReal* G
  ) const
{
  auto iX = X.derived( ).template cast< TReal >( );
  auto iY = Y.derived( ).template cast< TReal >( );

  TCol Z;
  this->m_Model->evaluate( Z, iX );
  Z -= iY;

  if( G != nullptr )
  {
    G[ 0 ] = Z.mean( );
    MRow( G + 1, 1, this->m_Model->number_of_inputs( ) ) =
      ( iX.array( ).colwise( ) * Z.array( ) ).colwise( ).mean( );
  } // end if
  return( ( ( Z.transpose( ) * Z ) / TReal( Z.rows( ) ) )( 0 ) );
}

#endif // __PUJ_ML__Model__Regression__Linear__hxx__

// eof - $RCSfile$
