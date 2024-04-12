// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__MeanSquareError__hxx__
#define __ivqML__Cost__MeanSquareError__hxx__

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Cost::MeanSquareError< _TModel >::
MeanSquareError( )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
typename ivqML::Cost::MeanSquareError< _TModel >::
TScl ivqML::Cost::MeanSquareError< _TModel >::
operator()( const TModel& model, TScl* G ) const
{
  TMat Z = model.eval( this->m_X ) - this->m_Y.array( );
  if( G != nullptr )
  {
    TNat n = this->m_X.rows( );

    *G = Z.mean( ) * TScl( 2 );
    TColMap( G + 1, n, 1 ) =
      ( this->m_X * Z.transpose( ) ) * ( TScl( 2 ) / TScl( n ) );
  } // end if
  return( Z.array( ).pow( 2 ).mean( ) );
}

#endif // __ivqML__Cost__MeanSquareError__hxx__

// eof - $RCSfile$
