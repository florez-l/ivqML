// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <cassert>
#include <PUJ/Regression.h>

// -------------------------------------------------------------------------
template< class _TScalar >
PUJ::Regression::MSE< _TScalar >::
MSE( const TMatrix& X, const TMatrix& y )
  : Superclass( X, y )
{
  TScalar m = TScalar( this->m_X.rows( ) );
  this->m_XtX = ( this->m_X.transpose( ) * this->m_X ) / m;

  this->m_Xby = ( this->m_X.array( ) * this->m_y.array( ) ).colwise( ).mean( );
  this->m_uX = this->m_X.colwise( ).mean( );
  this->m_uy = this->m_y.mean( );
  this->m_yty = ( this->m_y.transpose( ) * this->m_y )( 0, 0 ) / m;
}

// -------------------------------------------------------------------------
template< class _TScalar >
void PUJ::Regression::MSE< _TScalar >::
AnalyticSolve( TRowVector& theta )
{
  /* TODO
     x = numpy.append( numpy.array( [ self.m_uy ] ), self.m_Xby, axis = 0 )
     B = numpy.append( numpy.matrix( [ 1 ] ), self.m_uX, axis = 1 )
     A = numpy.append( self.m_uX.T, self.m_XtX, axis = 1 )
     return x @ numpy.linalg.inv( numpy.append( B, A, axis = 0 ) )
  */
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename PUJ::Regression::MSE< _TScalar >::
TScalar PUJ::Regression::MSE< _TScalar >::
CostAndGradient( TRowVector& gt, const TRowVector& theta )
{
  static const TScalar _2 = TScalar( 2 );
  TScalar b = theta( 0, 0 );
  TRowVector w = theta.block( 0, 1, 0, theta.cols( ) );

  TScalar J =
    ( w * this->m_XtX * w.transpose( ) )( 0, 0 ) +
    ( ( w * this->m_uX.transpose( ) )( 0, 0 ) * ( _2 * b ) ) +
    ( b * b ) -
    ( ( w * this->m_Xby.transpose( ) )( 0, 0 ) * _2 ) -
    ( this->m_uy * _2 * b ) +
    this->m_yty;

  gt( 0, 0 ) =
    ( ( w * this->m_uX.transpose( ) )( 0, 0 ) + b - this->m_uy ) * _2;
  gt.block( 0, 1, 0, gt.cols( ) ) =
    ( ( w * this->m_XtX ) + ( this->m_uX * b ) - this->m_Xby ) * _2;

  return( J );
}

// -------------------------------------------------------------------------
#include <PUJ_ML_export.h>
template class PUJ_ML_EXPORT PUJ::Regression::MSE< float >;
template class PUJ_ML_EXPORT PUJ::Regression::MSE< double >;
template class PUJ_ML_EXPORT PUJ::Regression::MSE< long double >;

// eof - $RCSfile$
