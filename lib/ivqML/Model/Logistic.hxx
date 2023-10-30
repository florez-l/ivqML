// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Logistic__hxx__
#define __ivqML__Model__Logistic__hxx__

#include <cmath>
#include <limits>

// -------------------------------------------------------------------------
template< class _S >
template< class _Y, class _X >
void ivqML::Model::Logistic< _S >::
operator()(
  Eigen::EigenBase< _Y >& iY, const Eigen::EigenBase< _X >& iX,
  bool derivative
  ) const
{
  using _YS = typename _Y::Scalar;
  using _YM = Eigen::Matrix< _YS, Eigen::Dynamic, Eigen::Dynamic >;
  static const _YS _0 = _YS( 0 );
  static const _YS _1 = _YS( 1 );
  static const _YS _E = std::numeric_limits< _YS >::epsilon( );
  static const _YS _L = std::log( _1 - _E ) - std::log( _E );
  static const auto f = [&]( _YS z ) -> _YS
    {
      if     ( z >  _L ) return( _1 );
      else if( z < -_L ) return( _0 );
      else               return( _1 / ( _1 + std::exp( -z ) ) );
    };
  static const auto d = [&]( _YS z ) -> _YS
    {
      _YS s = f( z );
      return( s * ( _1 - s ) );
    };

  if( derivative )
  {
    _YM Z;
    this->Superclass::operator()( Z, iX, false );
    this->Superclass::operator()( iY, iX, true );
    Z = Z.unaryExpr( d );
    iY.derived( ).array( ).colwise( ) *= Z.col( 0 ).array( );
  }
  else
  {
    this->Superclass::operator()( iY, iX, false );
    iY.derived( ) = iY.derived( ).unaryExpr( f );
  } // end if
}

// -------------------------------------------------------------------------
template< class _S >
template< class _Y, class _X >
void ivqML::Model::Logistic< _S >::
threshold(
  Eigen::EigenBase< _Y >& iY, const Eigen::EigenBase< _X >& iX
  ) const
{
  using _YS = typename _Y::Scalar;
  using _YM = Eigen::Matrix< _YS, Eigen::Dynamic, Eigen::Dynamic >;
  static const _YS _0 = _YS( 0 );
  static const _YS _1 = _YS( 1 );
  static const _YS _T = _YS( 0.5 );
  static const auto t = [&]( _YS z ) -> _YS
    {
      return( ( z < _T )? _0: _1 );
    };
  this->operator()( iY, iX, false );
  iY.derived( ) = iY.derived( ).unaryExpr( t );
}

#endif // __ivqML__Model__Logistic__hxx__

// eof - $RCSfile$
