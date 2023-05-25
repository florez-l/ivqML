// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizers__GradientDescent__hxx__
#define __PUJ_ML__Optimizers__GradientDescent__hxx__

// -------------------------------------------------------------------------
template< class _C, class _X, class _Y >
PUJ_ML::Optimizers::GradientDescent< _C, _X, _Y >::
GradientDescent( TModel& m, const TX& X, const TY& Y )
  : Superclass( m, X, Y )
{
}

// -------------------------------------------------------------------------
template< class _C, class _X, class _Y >
void PUJ_ML::Optimizers::GradientDescent< _C, _X, _Y >::
fit( )
{
  auto iX = this->m_X->derived( ).template cast< TReal >( );
  auto iY = this->m_Y->derived( ).template cast< TReal >( );

  this->m_Model->init( iX.cols( ) );

  TCost cost( this->m_Model );
  TRow G( this->m_Model->number_of_parameters( ) );
  std::vector< std::vector< unsigned long long > > batches;
  this->_batches( batches );

  unsigned long long epoch = 0;
  bool stop = false;
  TReal J, mG;
  while( !stop )
  {
    // Update gradient
    for( const auto& batch: batches )
      J = cost.evaluate(
        iX( batch, Eigen::placeholders::all ),
        iY( batch, Eigen::placeholders::all ),
        G.data( )
        );
    mG = G * G.transpose( );

    // Descent
    this->m_Model->move_parameters( G.data( ), -this->m_Alpha );

    // Check stopping criteria
    epoch++;
    stop |= !( epoch < this->m_MaxEpochs && this->m_Epsilon < mG );

    // Show debug information and, possibly, stop.
    if( epoch % this->m_DebugEpochs == 0 || epoch == 1 )
      stop |= this->m_Debug( J, mG, epoch );
  } // end while
  this->m_Debug( J, mG, epoch );
}

#endif // __PUJ_ML__Optimizers__GradientDescent__hxx__

// eof - $RCSfile$
