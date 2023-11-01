// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__GradientDescent__hxx__
#define __ivqML__Optimizer__GradientDescent__hxx__

// -------------------------------------------------------------------------
template< class _C >
ivqML::Optimizer::GradientDescent< _C >::
GradientDescent( TModel& m, const TX& iX, const TY& iY )
  : Superclass( m, iX, iY )
{
  this->m_P.add_options( )
    ivqML_Optimizer_OptionMacro( alpha, "alpha,a" );
}

// -------------------------------------------------------------------------
template< class _C >
void ivqML::Optimizer::GradientDescent< _C >::
fit( )
{
  TScalar e =
    std::pow( TScalar( 10 ), std::log10( this->m_alpha ) * TScalar( 2 ) );

  // Cost function
  _C cost( *( this->m_M ), *( this->m_X ), *( this->m_Y ) );
  auto J = cost( );
  auto G = TConstMap( J.second, 1, this->m_M->number_of_parameters( ) );

  // Main loop
  bool stop = false, debug_stop = false;
  TNatural i = 0;
  while( !stop )
  {
    *( this->m_M ) -= G * this->m_alpha;
    if( i % this->m_debug_iterations == 0 )
      debug_stop = this->m_D( J.first, G.norm( ), this->m_M, i );
    i++;
    stop =
      ( G.norm( ) <= e )
      ||
      ( this->m_max_iterations == i )
      ||
      debug_stop;
    if( !stop )
      J = cost( );
  } // end while
  this->m_D( J.first, G.norm( ), this->m_M, i );
}

#endif // __ivqML__Optimizer__GradientDescent__hxx__

// eof - $RCSfile$
