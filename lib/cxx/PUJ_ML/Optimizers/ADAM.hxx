// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizers__ADAM__hxx__
#define __PUJ_ML__Optimizers__ADAM__hxx__

// -------------------------------------------------------------------------
template< class _C, class _X, class _Y >
PUJ_ML::Optimizers::ADAM< _C, _X, _Y >::
ADAM( TModel& m, const TX& X, const TY& Y )
  : Superclass( m, X, Y )
{
  this->m_SlideEpsilon = this->m_Epsilon;
}

// -------------------------------------------------------------------------
template< class _C, class _X, class _Y >
void PUJ_ML::Optimizers::ADAM< _C, _X, _Y >::
fit( )
{
  static const TReal _1 = TReal( 1 );
  static const TReal _2 = TReal( 2 );

  auto iX = this->m_X->derived( ).template cast< TReal >( );
  auto iY = this->m_Y->derived( ).template cast< TReal >( );

  this->m_Model->init( iX.cols( ) );

  TCost cost( this->m_Model );
  std::vector< std::vector< unsigned long long > > batches;
  this->_batches( batches );

  TReal J, mG;
  TReal b1 = this->m_Beta1;
  TReal b2 = this->m_Beta2;
  TReal b1t = this->m_Beta1;
  TReal b2t = this->m_Beta2;
  std::vector< TReal > G( this->m_Model->number_of_parameters( ), 0 );
  G.shrink_to_fit( );
  MCol g( G.data( ), G.size( ), 1 );
  unsigned long long epoch = 0;
  bool stop = false;
  TCol m = TCol::Zero( G.size( ) );
  TCol v = TCol::Zero( G.size( ) );

  while( !stop )
  {
    for( const auto& batch: batches )
    {
      // Gradient
      J = cost.gradient( G, iX( batch, Eigen::all ), iY( batch, Eigen::all ) );
      mG = g.transpose( ) * g;

      // Modify gradient with moments
      m = ( m * b1 ) + ( g * ( _1 - b1 ) );
      v = ( v * b2 ) + TCol( g.array( ).pow( _2 ) * ( _1 - b2 ) );
      g =
        ( m / ( _1 - b1t ) ).array( )
        /
        ( ( v / ( _1 - b2t ) ).array( ).sqrt( ) + this->m_SlideEpsilon );

      // Descent
      this->m_Model->move_parameters( G, -this->m_Alpha );
    } // end for

    // Next epoch
    b1t *= b1;
    b2t *= b2;
    epoch++;

    // Check stopping criteria
    stop |= !( epoch < this->m_MaxEpochs && this->m_Epsilon < mG );

    // Show debug information and, possibly, stop.
    if( epoch % this->m_DebugEpochs == 0 || epoch == 1 )
      stop |= this->m_Debug( J, mG, epoch );
  } // end while
  this->m_Debug( J, mG, epoch );
}

#endif // __PUJ_ML__Optimizers__ADAM__hxx__

// eof - $RCSfile$
