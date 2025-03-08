// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Adam__hxx__
#define __ivqML__Optimizer__Adam__hxx__

#include <cmath>

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Optimizer::Adam< _TModel >::
Adam( TModel& m )
  : Superclass( m )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Optimizer::Adam< _TModel >::
~Adam( )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
const typename ivqML::Optimizer::Adam< _TModel >::
TReal& ivqML::Optimizer::Adam< _TModel >::
beta1( ) const
{
  return( this->m_Beta1 );
}

// -------------------------------------------------------------------------
template< class _TModel >
const typename ivqML::Optimizer::Adam< _TModel >::
TReal& ivqML::Optimizer::Adam< _TModel >::
beta2( ) const
{
  return( this->m_Beta2 );
}

// -------------------------------------------------------------------------
template< class _TModel >
void ivqML::Optimizer::Adam< _TModel >::
setBeta1( const TReal& b )
{
  this->m_Beta1 = b;
}

// -------------------------------------------------------------------------
template< class _TModel >
void ivqML::Optimizer::Adam< _TModel >::
setBeta2( const TReal& b )
{
  this->m_Beta2 = b;
}

// -------------------------------------------------------------------------
template< class _TModel >
template< class _TX_tr, class _Ty_tr, class _TX_te, class _Ty_te >
void ivqML::Optimizer::Adam< _TModel >::
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

  TReal e = this->m_Epsilon;
  TReal b1 = this->m_Beta1;
  TReal b2 = this->m_Beta2;
  TReal cb1 = _1 - b1;
  TReal cb2 = _1 - b2;
  TReal b1t = b1;
  TReal b2t = b2;
  TNatural t = 0;
  bool stop = false;
  TCol G( this->m_Model->size( ) );
  TCol mt = TCol::Zero( G.rows( ) );
  TCol vt = TCol::Zero( G.rows( ) );

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
      {
        TReal i1 = _1 / ( _1 - b1t );
        TReal i2 = _1 / ( _1 - b2t );

        mt = ( mt * b1 ) + ( G * cb1 );
        vt = ( vt * b2 ) + ( G.array( ).pow( 2 ) * cb2 ).matrix( );
        auto D =
          ( mt * i1 ).array( ) / ( ( ( vt * i2 ).array( ) ).sqrt( ) + e );
        
        *( this->m_Model ) -= D.matrix( ) * this->m_Alpha;

        b1t *= b1;
        b2t *= b2;
      } // end if
    }
    else
      stop = true;
  } // end while
}

#endif // __ivqML__Optimizer__Adam__hxx__

// eof - $RCSfile$
