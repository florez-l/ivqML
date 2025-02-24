// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__GradientDescent__hxx__
#define __ivqML__Optimizer__GradientDescent__hxx__

#include <cmath>

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Optimizer::GradientDescent< _TModel >::
GradientDescent( TModel& m )
  : Superclass( m )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Optimizer::GradientDescent< _TModel >::
~GradientDescent( )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
const typename ivqML::Optimizer::GradientDescent< _TModel >::
TReal& ivqML::Optimizer::GradientDescent< _TModel >::
alpha( ) const
{
  return( this->m_Alpha );
}

// -------------------------------------------------------------------------
template< class _TModel >
void ivqML::Optimizer::GradientDescent< _TModel >::
setAlpha( const TReal& a )
{
  this->m_Alpha = a;
}

// -------------------------------------------------------------------------
template< class _TModel >
template< class _TX_tr, class _Ty_tr, class _TX_te, class _Ty_te >
void ivqML::Optimizer::GradientDescent< _TModel >::
fit(
  const Eigen::EigenBase< _TX_tr >& bX_train,
  const Eigen::EigenBase< _Ty_tr >& by_train,
  const Eigen::EigenBase< _TX_te >& bX_test,
  const Eigen::EigenBase< _Ty_te >& by_test
  )
{
  static const TReal _0 = TReal( 0 );
  static const TReal _1 = TReal( 1 );
  static const TReal _M = std::numeric_limits< TReal >::max( );

  auto X_tr = bX_train.derived( ).template cast< TReal >( );
  auto y_tr = by_train.derived( ).template cast< TReal >( );
  auto X_te = bX_test.derived( ).template cast< TReal >( );
  auto y_te = by_test.derived( ).template cast< TReal >( );

  TNatural t = 0;
  bool stop = false;
  TColumn G( this->m_Model->size( ) );

  while( !stop )
  {
    t++;

    TReal J_tr
      =
      this->m_Model->
      cost_gradient( G, X_tr, y_tr, this->m_Lambda1, this->m_Lambda2 );
    if( !std::isnan( J_tr ) && !std::isinf( J_tr ) )
    {
      TReal J_te
        =
        ( 0 < X_te.rows( ) )? this->m_Model->cost( X_te, y_te ): _M;

      stop
        =
        this->m_Debug(
          t, std::sqrt( G.array( ).pow( 2 ).sum( ) ), J_tr, J_te
          );

      if( !stop )
        *( this->m_Model ) -= G * this->m_Alpha;
    }
    else
      stop = true;
  } // end while
}

#endif // __ivqML__Optimizer__GradientDescent__hxx__

// eof - $RCSfile$
