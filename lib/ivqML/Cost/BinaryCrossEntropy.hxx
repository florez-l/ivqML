// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__BinaryCrossEntropy__hxx__
#define __ivqML__Cost__BinaryCrossEntropy__hxx__

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Cost::BinaryCrossEntropy< _TModel >::
BinaryCrossEntropy( )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
typename ivqML::Cost::BinaryCrossEntropy< _TModel >::
TScl ivqML::Cost::BinaryCrossEntropy< _TModel >::
operator()( const TModel& model, TScl* G ) const
{
  /* TODO
     static const TScl _E = std::numeric_limits< TScl >::epsilon( );

     TNat m = this->m_X.cols( );
     TMat Z = this->m_M->eval( this->m_X );
     std::atomic< TScl > S = 0;
     Z.noalias( )
     =
     Z.NullaryExpr(
     Z.rows( ), Z.cols( ),
     [&]( const Eigen::Index& r, const Eigen::Index& c ) -> TScl
     {
     TScl z = Z( r, c );
     TScl l = ( this->m_Y( r, c ) == 0 )? ( TScl( 1 ) - z ): z;
     S = S - ( std::log( ( _E < l )? l: _E ) / TScl( m ) );
     return( z - this->m_Y( r, c ) );
     }
     );

     if( G != nullptr )
     {
     if( this->m_M->has_backpropagation( ) )
     {
     if( this->m_B == nullptr )
     this->m_B =
     reinterpret_cast< TScl* >(
     std::calloc(
     this->m_M->buffer_size( ) * m, sizeof( TScl )
     )
     );
     this->m_M->backpropagation( G, this->m_B, this->m_X, this->m_Y );
     }
     else
     {
     *G = Z.mean( );
     TColMap( G + 1, this->m_X.rows( ), 1 )
     =
     ( this->m_X * Z.transpose( ) ) / TScl( m );
     } // end if
     } // end if
     return( TScl( S ) );
  */
  return( TScl( 0 ) );
}

#endif // __ivqML__Cost__BinaryCrossEntropy__hxx__

// eof - $RCSfile$
