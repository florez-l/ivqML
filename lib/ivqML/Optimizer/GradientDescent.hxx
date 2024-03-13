// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__GradientDescent__hxx__
#define __ivqML__Optimizer__GradientDescent__hxx__

// -------------------------------------------------------------------------
template< class _TCost >
ivqML::Optimizer::GradientDescent< _TCost >::
GradientDescent( )
  : Superclass( )
{
  this->m_Options.add_options( )
    ivqML_Optimizer_OptionMacro( learning_rate, "alpha,A" );
}

// -------------------------------------------------------------------------
template< class _TCost >
void ivqML::Optimizer::GradientDescent< _TCost >::
fit( )
{
  if( !this->has_model( ) )
    return;

  // Initialize optimizer
  TNatural p = this->m_Model->number_of_parameters( );
  TRowMap mp = this->m_Model->row( p );
  TRow G( p );
  bool stop = false;
  TNatural i = 0;

  // Main loop
  while( !stop )
  {
    // Update function
    TScalar J = this->m_Costs[ 0 ]( G.data( ) );
    mp -= G * this->m_learning_rate;

    // Check stop
    TScalar gn = G.norm( );
    stop  = ( gn < this->m_epsilon );
    stop |= ( std::isnan( gn ) || std::isinf( gn ) );
    stop |= ( ++i >= this->m_max_iterations );

    // Process debug information
    stop |=
      this->m_Debugger(
        J, gn, this->m_Model, i,
        stop || i == 1 || i % this->m_debug_iterations == 0
        );
  } // end while
}

#endif // __ivqML__Optimizer__GradientDescent__hxx__

// eof - $RCSfile$
