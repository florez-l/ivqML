// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Cost__hxx__
#define __ivqML__Model__Cost__hxx__

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TY, class _TZ >
typename ivqML::Model::Cost< _TReal >::
TReal ivqML::Model::Cost< _TReal >::
operator()(
  const Eigen::EigenBase< _TY >& bY, const Eigen::EigenBase< _TZ >& bZ
  ) const
{
  auto Y = bY.derived( ).template cast< TReal >( );
  auto Z = bZ.derived( ).template cast< TReal >( );

  if( this->m_Type == Self::MCE )
    return( this->_mce( Y, Z ) );
  else if( this->m_Type == Self::CCE )
    return( this->_cce( Y, Z ) );
  else // if( this->m_Type == Self::MSE )
    return( this->_mse( Y, Z ) );
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TY, class _TZ >
typename ivqML::Model::Cost< _TReal >::
TReal ivqML::Model::Cost< _TReal >::
_mse(
  const Eigen::EigenBase< _TY >& Y, const Eigen::EigenBase< _TZ >& Z
  ) const
{
  return( ( Y.derived( ) - Z.derived( ) ).array( ).pow( 2 ).mean( ) );
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TY, class _TZ >
typename ivqML::Model::Cost< _TReal >::
TReal ivqML::Model::Cost< _TReal >::
_mce(
  const Eigen::EigenBase< _TY >& Y, const Eigen::EigenBase< _TZ >& Z
  ) const
{
  static const TReal E
    =
    std::pow(
      TReal( 10 ),
      std::log10( std::numeric_limits< TReal >::epsilon( ) ) * TReal( 0.5 )
      );
  static const TReal D = std::log( E );

  return(
    Y.derived( ).binaryExpr(
      Z.derived( ),
      [&]( const TReal& y, const TReal& z ) -> TReal
      {
        TReal zz = ( y == TReal( 1 ) )? z: TReal( 1 ) - z;
        return( -( ( E < zz )? std::log( zz ): D ) );
      }
      ).mean( )
    );
}

// -------------------------------------------------------------------------
template< class _TReal >
template< class _TY, class _TZ >
typename ivqML::Model::Cost< _TReal >::
TReal ivqML::Model::Cost< _TReal >::
_cce(
  const Eigen::EigenBase< _TY >& Y, const Eigen::EigenBase< _TZ >& Z
  ) const
{
  static const TReal E
    =
    std::pow(
      TReal( 10 ),
      std::log10( std::numeric_limits< TReal >::epsilon( ) ) * TReal( 0.5 )
      );
  static const TReal D = std::log( E );

  return(
    Y.derived( ).binaryExpr(
      Z.derived( ),
      [&]( const TReal& y, const TReal& z ) -> TReal
      {
        if( y != TReal( 0 ) )
          return( -( ( E < z )? std::log( z ): D ) );
        else
          return( TReal( 0 ) );
      }
      ).mean( )
    );
}

#endif // __ivqML__Model__Cost__hxx__

// eof - $RCSfile$
