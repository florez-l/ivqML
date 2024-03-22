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
}

// -------------------------------------------------------------------------
template< class _TCost >
ivqML::Optimizer::GradientDescent< _TCost >::
~GradientDescent( )
{
}

// -------------------------------------------------------------------------
template< class _TCost >
void ivqML::Optimizer::GradientDescent< _TCost >::
register_options( boost::program_options::options_description& opt )
{
  this->Superclass::register_options( opt );
  opt.add_options( )
    ivqML_Optimizer_OptionMacro( alpha, "learning_rate,a" );
}

// -------------------------------------------------------------------------
template< class _TCost >
void ivqML::Optimizer::GradientDescent< _TCost >::
fit( TModel& model )
{
  // Associate model to every batch
  for( TCost& cost: this->m_Costs )
    cost.set_model( &model );

  // Initialize
  TNat p = model.number_of_parameters( );
  TRowMap mp = model.row( p );
  TRow G( p ), D( p );
  bool stop = false;
  TNat i = 0;

  // Main loop
  while( !stop )
  {
    // Update function
    for( TNat c = 0; c < this->m_Costs.size( ); ++c )
    {
      this->m_Costs[ c ]( G.data( ) );
      mp -= G * this->m_alpha;

      if( c == 0 ) D  = G;
      else         D += G;
    } // end for

    // Check stop
    TScl dn = D.norm( );
    stop  = ( dn < this->m_epsilon );
    stop |= ( std::isnan( dn ) || std::isinf( dn ) );
    stop |= ( ++i >= this->m_max_iter );
    /* TODO
       stop |= this->m_Debugger( &model, dn, i );
    */

    /* TODO
       std::cerr << i << " " << dn << " " << this->m_epsilon << std::endl;
    */

    // Process debug information
    /* TODO
       stop |=
       this->m_Debugger(
       J, dn, this->m_Model, i,
       stop || i == 1 || i % this->m_debug_iterations == 0
       );
    */
  } // end while
}

#endif // __ivqML__Optimizer__GradientDescent__hxx__

// eof - $RCSfile$
