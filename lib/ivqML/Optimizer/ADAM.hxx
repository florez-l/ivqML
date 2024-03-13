// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__ADAM__hxx__
#define __ivqML__Optimizer__ADAM__hxx__

// -------------------------------------------------------------------------
template< class _TCost >
ivqML::Optimizer::ADAM< _TCost >::
ADAM( )
  : Superclass( )
{
  this->m_Options.add_options( )
    ivqML_Optimizer_OptionMacro( alpha, "alpha,a" )
    ivqML_Optimizer_OptionMacro( beta1, "beta1,b" )
    ivqML_Optimizer_OptionMacro( beta2, "beta2,c" );
}

// -------------------------------------------------------------------------
template< class _TCost >
void ivqML::Optimizer::ADAM< _TCost >::
fit( )
{
  if( !this->has_model( ) )
    return;

  // Some values...
  static const TScalar _1 = TScalar( 1 );
  static const TScalar _2 = TScalar( 2 );
  static const TScalar _10 = TScalar( 10 );
  TScalar e = std::pow( _10, std::log10( this->m_alpha ) * _2 );
  TScalar b1t = this->m_beta1;
  TScalar b2t = this->m_beta2;
  TScalar b1 = _1 - this->m_beta1;
  TScalar b2 = _1 - this->m_beta2;

  // Initialize optimizer
  TNatural p = this->m_Model->number_of_parameters( );
  TRowMap mp = this->m_Model->row( p );
  bool stop = false;
  TNatural i = 0;

  // Prepare loop
  TRow G( p );
  TRow M = TRow::Zero( p );
  TRow V = TRow::Zero( p );
  TRow M2 = M;
  TRow V2 = V;

  // Main loop
  while( !stop )
  {
    // Update function
    M *= this->m_beta1;
    V *= this->m_beta2;
    TScalar J = this->m_Costs[ 0 ]( G.data( ) );
    M2 = ( M + ( G * b1 ) ) / ( _1 - b1t );
    V2 =
      ( ( V.array( ) + ( G.array( ).pow( 2 ) * b2 ) ) / ( _1 - b2t ) )
      .sqrt( );
    G.array( ) = M2.array( ) / ( V2.array( ) + e );
    mp -= G * this->m_alpha;
    M = M2;
    V = V2;
    b1t *= this->m_beta1;
    b2t *= this->m_beta2;

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

#endif // __ivqML__Optimizer__ADAM__hxx__

// eof - $RCSfile$
