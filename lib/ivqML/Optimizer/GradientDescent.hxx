// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__GradientDescent__hxx__
#define __ivqML__Optimizer__GradientDescent__hxx__

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Optimizer::GradientDescent< _TModel >::
GradientDescent( )
  : Superclass( )
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
void ivqML::Optimizer::GradientDescent< _TModel >::
_fit( TModel& model, TBatches& batches, TShuffler shuffler )
{
  Eigen::Map< const TMatrix > X( this->m_X, model.input_size( ), this->m_M );
  Eigen::Map< const TMatrix > Y( this->m_Y, model.output_size( ), this->m_M );

  TRow G( model.size( ) );
  TRow sG( model.size( ) );

  TReal J = std::numeric_limits< TReal >::max( );
  TNatural t = 0;
  bool stop = false;
  while( !stop )
  {
    t++;

    shuffler( );

    J = 0;
    sG.fill( TReal( 0 ) );
    for( const TBatch& b: batches )
    {
      J += model.gradient(
        G.data( ), X( Eigen::all, b ), Y( Eigen::all, b ),
        this->m_Lambda1, this->m_Lambda2
        );
      model -= G * this->m_Alpha;
      sG += G;
    } // end for
    J /= TReal( batches.size( ) );

    // Check stop
    stop
      =
      this->m_Debugger(
        t, model, J, sG * sG.transpose( ), this->m_X, this->m_Y, this->m_M
        );
  } // end while
  this->m_Debugger(
    t, model, J, sG * sG.transpose( ), this->m_X, this->m_Y, this->m_M
    );
}

#endif // __ivqML__Optimizer__GradientDescent__hxx__

// eof - $RCSfile$
