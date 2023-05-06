// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Linear__hxx__
#define __PUJ_ML__Model__Linear__hxx__

#include <Eigen/Dense>

// -------------------------------------------------------------------------
template< class _R >
PUJ_ML::Model::Linear< _R >::
Linear( const unsigned long long& n )
  : Superclass( n )
{
}

// -------------------------------------------------------------------------
template< class _R >
PUJ_ML::Model::Linear< _R >::
~Linear( )
{
  if( this->m_T != nullptr )
    delete this->m_T;
}

// -------------------------------------------------------------------------
template< class _R >
unsigned long long PUJ_ML::Model::Linear< _R >::
number_of_inputs( ) const
{
  return( this->m_P.size( ) - 1 );
}

// -------------------------------------------------------------------------
template< class _R >
void PUJ_ML::Model::Linear< _R >::
set_number_of_parameters( const unsigned long long& n )
{
  this->Superclass::set_number_of_parameters( n + 1 );
  this->m_T = new MCol( this->m_P.data( ) + 1, this->m_P.size( ) - 1, 1 );
}

// -------------------------------------------------------------------------
template< class _R >
template< class _Y, class _X >
void PUJ_ML::Model::Linear< _R >::
evaluate( Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X ) const
{
  Y.derived( ) =
    (
      ( X.derived( ).template cast< _R >( ) * *( this->m_T ) ).array( )
      +
      this->m_P[ 0 ]
      )
    .template cast< typename _Y::Scalar >( );
}

// -------------------------------------------------------------------------
template< class _R >
template< class _Y, class _X >
void PUJ_ML::Model::Linear< _R >::
fit( const Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X )
{
  /* TODO
     unsigned long long m = X.rows( );
     unsigned long long n = X.cols( );
     this->set_number_of_parameters( n );

     auto iX = X.derived( ).template cast< _R >( );
     auto iY = Y.derived( ).template cast< _R >( );

     TMatrix b( 1, n + 1 );
     b( 0, 0 ) = iY.mean( );
     b.block( 0, 1, 1, n ) = ( iX.array( ) * iY.array( ) ).colwise( ).mean( );

     TMatrix M( n + 1, n + 1 );
     M( 0, 0 ) = _R( 1 );
     M.block( 0, 1, 1, n ) = iX.colwise( ).mean( );
     M.block( 1, 0, n, 1 ) = M.block( 0, 1, 1, n ).transpose( );
     M.block( 1, 1, n, n ) = ( iX.transpose( ) * iX ).array( ) / _R( m );

     std::cout << "--------------------" << std::endl;
     std::cout << M << std::endl;
     std::cout << "--------------------" << std::endl;
     std::cout << M.inverse( ).transpose( ) * b.transpose( ) << std::endl;
     std::cout << "--------------------" << std::endl;

     Eigen::Map< TRow >( this->m_P.data( ), 1, n + 1 ) = b * M.inverse( );
  */
}

// -------------------------------------------------------------------------
template< class _R >
PUJ_ML::Model::Linear< _R >::Cost_MSE::
Cost_MSE( TModel* m )
  : m_Model( m )
{
}

// -------------------------------------------------------------------------
template< class _R >
template< class _X, class _Y >
typename PUJ_ML::Model::Linear< _R >::
TReal PUJ_ML::Model::Linear< _R >::Cost_MSE::
evaluate(
  const Eigen::EigenBase< _X >& X,
  const Eigen::EigenBase< _Y >& Y
  ) const
{
  auto iX = X.derived( ).template cast< TReal >( );
  auto iY = Y.derived( ).template cast< TReal >( );

  TCol Z;
  this->m_Model->evaluate( Z, iX );
  Z -= iY;
  return( ( ( Z.transpose( ) * Z ) / TReal( Z.rows( ) ) )( 0 ) );
}

// -------------------------------------------------------------------------
template< class _R >
template< class _X, class _Y >
typename PUJ_ML::Model::Linear< _R >::
TReal PUJ_ML::Model::Linear< _R >::Cost_MSE::
gradient(
  std::vector< TReal >& G,
  const Eigen::EigenBase< _X >& X,
  const Eigen::EigenBase< _Y >& Y
  ) const
{
  if( G.size( ) != this->m_Model->number_of_parameters( ) )
  {
    G.resize( this->m_Model->number_of_parameters( ), 0 );
    G.shrink_to_fit( );
  } // end if

  auto iX = X.derived( ).template cast< TReal >( );
  auto iY = Y.derived( ).template cast< TReal >( );

  TCol Z;
  this->m_Model->evaluate( Z, iX );
  Z -= iY;

  G[ 0 ] = Z.mean( );
  MRow( G.data( ) + 1, 1, G.size( ) - 1 ) =
    ( iX.array( ).colwise( ) * Z.array( ) ).colwise( ).mean( );
  return( ( ( Z.transpose( ) * Z ) / TReal( Z.rows( ) ) )( 0 ) );
}

#endif // __PUJ_ML__Model__Linear__hxx__

// eof - $RCSfile$
