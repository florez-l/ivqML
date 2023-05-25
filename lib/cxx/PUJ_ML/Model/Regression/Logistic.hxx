// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Regression__Logistic__hxx__
#define __PUJ_ML__Model__Regression__Logistic__hxx__

#include <algorithm>
#include <Eigen/Dense>

// -------------------------------------------------------------------------
template< class _R >
PUJ_ML::Model::Regression::Logistic< _R >::
Logistic( const unsigned long long& n )
  : Superclass( n )
{
}

// -------------------------------------------------------------------------
template< class _R >
template< class _Y, class _X >
void PUJ_ML::Model::Regression::Logistic< _R >::
evaluate( Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X ) const
{
  using _S = typename _Y::Scalar;
  static const _S _0 = _S( 0 );
  static const _S _1 = _S( 1 );
  static const _S _B = _S( 40 );
  static const auto f = [&]( _S z ) -> _S
    {
      if     ( z >  _B ) return( _1 );
      else if( z < -_B ) return( _0 );
      else               return( _1 / ( _1 + std::exp( -z ) ) );
    };

  this->Superclass::evaluate( Y, X );
  Y.derived( ) = Y.derived( ).unaryExpr( f );
}

// -------------------------------------------------------------------------
template< class _R >
template< class _Y, class _X >
void PUJ_ML::Model::Regression::Logistic< _R >::
threshold( Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X ) const
{
  static const auto f = []( TReal z ) -> typename _Y::Scalar
    {
      return( ( typename _Y::Scalar )( ( z < 0.5 )? 0: 1 ) );
    };

  TCol Z;
  this->Superclass::evaluate( Z, X );
  Y.derived( ) = Z.unaryExpr( f );
}

// -------------------------------------------------------------------------
template< class _R >
PUJ_ML::Model::Regression::Logistic< _R >::Cost::
Cost( TModel* m )
  : m_Model( m )
{
}

// -------------------------------------------------------------------------
template< class _R >
template< class _X, class _Y >
typename PUJ_ML::Model::Regression::Logistic< _R >::
TReal PUJ_ML::Model::Regression::Logistic< _R >::Cost::
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

  struct _V
  {
    _V( const TCol& Z )
      {
        this->Z = &Z;
        this->E =
          std::pow(
            TReal( 10 ),
            std::log10( std::numeric_limits< TReal >::epsilon( ) )
            *
            TReal( 0.5 )
            );
        this->D = std::log( this->E );
      }
    void init(
      const typename _Y::Scalar& y,
      const Eigen::Index& i,
      const Eigen::Index& j
      )
      {
        this->J = TReal( 0 );
        this->operator()( y, i, j );
      }
    void operator()(
      const typename _Y::Scalar& y,
      const Eigen::Index& i,
      const Eigen::Index& j
      )
      {
        TReal lv = this->Z->operator()( i );
        if( y == ( typename _Y::Scalar )( 0 ) )
          lv = TReal( 1 ) - lv;
        this->J -= ( this->E < lv )? std::log( lv ): this->D;
      }
    const TCol* Z;
    TReal J;
    TReal E;
    TReal D;
  } visitor( Z );
  iY.visit( visitor );

  if( G != nullptr )
  {
    G[ 0 ] = Z.mean( ) - iY.mean( );
    MRow( G + 1, 1, this->m_Model->number_of_inputs( ) ) =
      ( iX.array( ).colwise( ) * Z.array( ) ).colwise( ).mean( )
      -
      ( iX.array( ).colwise( ) * iY.col( 0 ).array( ) ).colwise( ).mean( );
  } // end if

  return( visitor.J / TReal( Z.rows( ) ) );
}

#endif // __PUJ_ML__Model__Regression__Logistic__hxx__

// eof - $RCSfile$
