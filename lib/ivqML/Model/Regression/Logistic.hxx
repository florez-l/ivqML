// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Regression__Logistic__hxx__
#define __ivqML__Model__Regression__Logistic__hxx__

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX >
auto ivqML::Model::Regression::Logistic< _TReal, _TNatural >::
operator()( const Eigen::EigenBase< _TX >& X, bool threshold ) const
{
  static const TReal _0  = TReal( 0 );
  static const TReal _05 = TReal( 0.5 );
  static const TReal _1  = TReal( 1 );
  static const TReal _M  = std::numeric_limits< TReal >::max( );
  static const TReal _L  = std::log( _M ) / TReal( 2 );
  auto f = [&]( TReal z ) -> TReal
    {
      if     ( z >  _L ) return( _1 );
      else if( z < -_L ) return( _0 );
      else
      {
        TReal s = _1 / ( _1 + std::exp( -z ) );
        return( ( threshold )? ( ( s < _05 )? _0: _1 ): s );
      } // end if
    };

  return( this->Superclass::operator()( X ).unaryExpr( f ) );
}

// -------------------------------------------------------------------------
/**
 * TODO: This method has no sense in a logistic regression
 */
template< class _TReal, class _TNatural >
template< class _TX, class _Ty >
void ivqML::Model::Regression::Logistic< _TReal, _TNatural >::
fit(
  const Eigen::EigenBase< _TX >& bX,
  const Eigen::EigenBase< _Ty >& by,
  const TReal& L1, const TReal& L2
  )
{
  /* TODO
     if( n == 0 || m != y.rows( ) )
     throw AssertionError( 'There is no closed solution for a logistic regression.' )
  */
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TG, class _TX, class _Ty >
typename ivqML::Model::Regression::Logistic< _TReal, _TNatural >::
TReal ivqML::Model::Regression::Logistic< _TReal, _TNatural >::
cost_gradient(
  Eigen::EigenBase< _TG >& G,
  const Eigen::EigenBase< _TX >& bX,
  const Eigen::EigenBase< _Ty >& by,
  const TReal& L1, const TReal& L2
  )
{
  auto X = bX.derived( ).template cast< TReal >( );
  auto y = by.derived( ).template cast< TReal >( );

  TColumn z = this->operator()( X );
  SVisitor v( z );
  y.visit( v );
  z -= y;

  G.derived( )( 0 , 0 ) = z.mean( );
  G.derived( ).block( 1, 0, X.cols( ), 1 )
    =
    ( X.array( ).colwise( ) * z.array( ) )
    .colwise( ).mean( ).transpose( );

  return( v.J / TReal( X.rows( ) ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TX, class _Ty >
typename ivqML::Model::Regression::Logistic< _TReal, _TNatural >::
TReal ivqML::Model::Regression::Logistic< _TReal, _TNatural >::
cost( const Eigen::EigenBase< _TX >& X, const Eigen::EigenBase< _Ty >& y )
{
  TColumn z = this->operator()( X );
  SVisitor v( z );
  y.derived( ).template cast< TReal >( ).visit( v );
  return( v.J / TReal( X.rows( ) ) );
}

#endif // __ivqML__Model__Regression__Logistic__hxx__

// eof - $RCSfile$
