// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__BinaryCrossEntropy__hxx__
#define __ivqML__Cost__BinaryCrossEntropy__hxx__

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Cost::BinaryCrossEntropy< _TModel >::
BinaryCrossEntropy( _TModel& m )
  : Superclass( m )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
typename ivqML::Cost::BinaryCrossEntropy< _TModel >::
TScalar ivqML::Cost::BinaryCrossEntropy< _TModel >::
operator()( TScalar* G ) const
{
  static const TScalar _E = std::numeric_limits< TScalar >::epsilon( );

  TScalar m = TScalar( this->m_X.cols( ) );
  TMatrix Z = this->m_M->evaluate( this->m_X );
  std::atomic< TScalar > S = 0;
  Z.noalias( )
    =
    Z.NullaryExpr(
      Z.rows( ), Z.cols( ),
      [&]( const Eigen::Index& r, const Eigen::Index& c ) -> TScalar
      {
        TScalar z = Z( r, c );
        TScalar l = ( this->m_Y( r, c ) == 0 )? ( TScalar( 1 ) - z ): z;
        S = S - ( std::log( ( _E < l )? l: _E ) / m );
        return( z - this->m_Y( r, c ) );
      }
      );

  if( G != nullptr )
  {
    *G = Z.mean( );
    Eigen::Map< TColumn >( G + 1, this->m_X.rows( ), 1 )
      =
      ( this->m_X * Z.transpose( ) ) / m;
  } // end if
  return( TScalar( S ) );
}

#endif // __ivqML__Cost__BinaryCrossEntropy__hxx__

// eof - $RCSfile$
