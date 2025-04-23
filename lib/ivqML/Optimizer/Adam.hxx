// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Adam__hxx__
#define __ivqML__Optimizer__Adam__hxx__

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Optimizer::Adam< _TModel >::
Adam( const TReal* Xtr, const TReal* Ytr, const TNatural& Mtr )
  : Superclass( Xtr, Ytr, Mtr )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Optimizer::Adam< _TModel >::
Adam(
  const TReal* Xtr, const TReal* Ytr,
  const TReal* Xte, const TReal* Yte,
  const TNatural& Mtr, const TNatural& Mte
  )
  : Superclass( Xtr, Ytr, Xte, Yte, Mtr, Mte )
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
set_beta1( const TReal& b )
{
  this->m_Beta1 = b;
}

// -------------------------------------------------------------------------
template< class _TModel >
void ivqML::Optimizer::Adam< _TModel >::
set_beta2( const TReal& b )
{
  this->m_Beta2 = b;
}

// -------------------------------------------------------------------------
template< class _TModel >
void ivqML::Optimizer::Adam< _TModel >::
_fit( TModel* model, const TBatches& batches )
{
  TNatural S = model->size( );
  TRow G = TRow::Zero( S );
  TRow M = TRow::Zero( S );
  TRow V = TRow::Zero( S );
  TRow sG = G;
  TReal e = this->m_Epsilon;
  TReal b1 = this->m_Beta1;
  TReal b2 = this->m_Beta2;
  TReal cb1 = TReal( 1 ) - b1;
  TReal cb2 = TReal( 1 ) - b2;
  TReal b1t = b1;
  TReal b2t = b2;

  bool stop = false;
  TNatural t = 0;
  while( !stop )
  {
    t++;

    TReal i1 = TReal( 1 ) / ( TReal( 1 ) - b1t );
    TReal i2 = TReal( 1 ) / ( TReal( 1 ) - b2t );
    sG.fill( 0 );
    TReal Jtr = 0;
    for( const TBatch& batch: batches )
    {
      Jtr
        +=
        model->gradient(
          G.data( ), batch.first, batch.second,
          this->m_Lambda1, this->m_Lambda2
          );
      M = ( M * b1 ) + ( G * cb1 );
      V = ( V * b2 ) + ( G.array( ).pow( 2 ) * cb2 ).matrix( );
      G = ( M * i1 ).array( ) / ( ( ( V * i2 ).array( ) ).sqrt( ) + e );

      sG += G;
      *model -= G * this->m_LearningRate;
    } // end for
    Jtr /= TReal( batches.size( ) );

    stop
      =
      this->m_Debugger(
        t, model, Jtr, sG * sG.transpose( ),
        this->m_Xtr, this->m_Ytr, this->m_Mtr,
        this->m_Xte, this->m_Yte, this->m_Mte
        );

    b1t *= b1;
    b2t *= b2;
  } // end while
}

#endif // __ivqML__Optimizer__Adam__hxx__

// eof - $RCSfile$
