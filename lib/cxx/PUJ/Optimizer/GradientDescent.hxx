// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Optimizer__GradientDescent__hxx__
#define __PUJ__Optimizer__GradientDescent__hxx__

// -------------------------------------------------------------------------
template< class _TModel >
PUJ::Optimizer::GradientDescent< _TModel >::
GradientDescent( TCost* cost )
  : m_Cost( cost )
{
  this->m_Debug =
    []( unsigned long long, TScalar, bool ) -> bool { return( false ); };
}

// -------------------------------------------------------------------------
template< class _TModel >
const unsigned long long& PUJ::Optimizer::GradientDescent< _TModel >::
GetRealIterations( ) const
{
  return( this->m_RealIterations );
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
SetDebug( TDebug d )
{
  this->m_Debug = d;
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
Fit( )
{
  TRow g;
  bool stop = false;
  TScalar J = ( *this->m_Cost )( &g );
  this->_Regularize( J, g );

  this->m_RealIterations = 0;
  while( !stop )
  {
    *this->m_Cost -= g * this->m_Alpha;
    TScalar Jn = ( *this->m_Cost )( &g );
    this->_Regularize( Jn, g );

    stop = ( ( J - Jn ) <= this->m_Epsilon );
    stop |=
      this->m_Debug(
        this->m_RealIterations, J,
        this->m_RealIterations % this->m_DebugIterations == 0
        );
    J = Jn;
    this->m_RealIterations++;
    stop |= ( this->m_MaximumNumberOfIterations <= this->m_RealIterations );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
_Regularize( TScalar& J, TRow& g )
{
  if( this->m_Lambda != TScalar( 0 ) )
  {
    const TRow& t = this->m_Cost->GetParameters( );
    J += t.squaredNorm( ) * this->m_Lambda;
    g += t * TScalar( 2 ) * this->m_Lambda;
  } // end if
}

#endif // __PUJ__Optimizer__GradientDescent__hxx__

// eof - $RCSfile$
