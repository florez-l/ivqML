// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__GradientDescent__hxx__
#define __ivqML__Optimizer__GradientDescent__hxx__

// -------------------------------------------------------------------------
template< class _C >
ivqML::Optimizer::GradientDescent< _C >::
GradientDescent( )
  : Superclass( )
{
  this->m_P.add_options( )
    ivqML_Optimizer_OptionMacro( alpha, "alpha,a" );
}

// -------------------------------------------------------------------------
template< class _C >
void ivqML::Optimizer::GradientDescent< _C >::
fit( )
{
  static const TScalar _2 = TScalar( 2 );
  static const TScalar _10 = TScalar( 10 );
  TScalar e = std::pow( _10, std::log10( this->m_alpha ) * _2 );

  // Cost function
  _C cost( *( this->m_M ), *( this->m_X ), *( this->m_Y ) );
  cost.set_batch_size( this->m_batch_size );
  auto J = cost( );
  auto G = TConstMap( J.second, 1, this->m_M->number_of_parameters( ) );
  TNatural B = cost.number_of_batches( );

  // Main loop
  bool stop = false, debug_stop = false;
  TNatural i = 0;
  while( !stop )
  {
    for( TNatural b = 0; b < B; ++b )
    {
      J = cost( b );
      *( this->m_M ) -= G * this->m_alpha;
    } // end for

    debug_stop =
      this->m_D(
        J.first, G.norm( ), this->m_M, i,
        ( i % this->m_debug_iterations == 0 )
        );
    i++;
    stop =
      ( G.norm( ) <= e )
      ||
      ( this->m_max_iterations == i )
      ||
      debug_stop;
  } // end while
  this->m_D( J.first, G.norm( ), this->m_M, i, true );
}

#endif // __ivqML__Optimizer__GradientDescent__hxx__

// eof - $RCSfile$
