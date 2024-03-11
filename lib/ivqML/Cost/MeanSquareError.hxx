// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__MeanSquareError__hxx__
#define __ivqML__Cost__MeanSquareError__hxx__

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Cost::MeanSquareError< _TModel >::
MeanSquareError( _TModel& m )
  : Superclass( m )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
typename ivqML::Cost::MeanSquareError< _TModel >::
TScalar ivqML::Cost::MeanSquareError< _TModel >::
operator()( TScalar* G ) const
{
  auto D = ( this->m_M->eval( this->m_X ) - this->m_Y.array( ) ).eval( );

  if( G != nullptr )
  {
    TNatural n = this->m_X.rows( );

    *G = D.mean( ) * TScalar( 2 );
    TColMap( G + 1, n, 1 )
      =
      ( this->m_X * D.matrix( ).transpose( ) )
      *
      ( TScalar( 2 ) / TScalar( n ) );
  } // end if
  return( D.array( ).pow( 2 ).mean( ) );
}

#endif // __ivqML__Cost__MeanSquareError__hxx__

// eof - $RCSfile$
