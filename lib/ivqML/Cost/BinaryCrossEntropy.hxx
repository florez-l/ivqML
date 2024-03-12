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

  TNatural m = this->m_X.cols( );
  TMat Z = this->m_M->eval( this->m_X );
  std::atomic< TScalar > S = 0;
  Z.noalias( )
    =
    Z.NullaryExpr(
      Z.rows( ), Z.cols( ),
      [&]( const Eigen::Index& r, const Eigen::Index& c ) -> TScalar
      {
        TScalar z = Z( r, c );
        TScalar l = ( this->m_Y( r, c ) == 0 )? ( TScalar( 1 ) - z ): z;
        S = S - ( std::log( ( _E < l )? l: _E ) / TScalar( m ) );
        return( z - this->m_Y( r, c ) );
      }
      );

  if( G != nullptr )
  {
    if( this->m_M->has_backpropagation( ) )
    {
      if( this->m_B == nullptr )
        this->m_B =
          reinterpret_cast< TScalar* >(
            std::calloc(
              this->m_M->buffer_size( ) * m, sizeof( TScalar )
              )
            );
      this->m_M->backpropagation( G, this->m_B, this->m_X, this->m_Y );
    }
    else
    {
      *G = Z.mean( );
      TColMap( G + 1, this->m_X.rows( ), 1 )
        =
        ( this->m_X * Z.transpose( ) ) / TScalar( m );
    } // end if
  } // end if
  return( TScalar( S ) );
}

#endif // __ivqML__Cost__BinaryCrossEntropy__hxx__

// eof - $RCSfile$
