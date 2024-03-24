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
}

// -------------------------------------------------------------------------
template< class _TCost >
ivqML::Optimizer::ADAM< _TCost >::
~ADAM( )
{
}

// -------------------------------------------------------------------------
template< class _TCost >
void ivqML::Optimizer::ADAM< _TCost >::
register_options( boost::program_options::options_description& opt )
{
  this->Superclass::register_options( opt );
  opt.add_options( )
    ivqML_Optimizer_OptionMacro( alpha, "learning_rate,a" )
    ivqML_Optimizer_OptionMacro( beta1, "beta1" )
    ivqML_Optimizer_OptionMacro( beta2, "beta2" );
}

// -------------------------------------------------------------------------
template< class _TCost >
void ivqML::Optimizer::ADAM< _TCost >::
fit( TModel& model )
{
  // Associate model to every batch
  /* TODO
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


     // Check stop
     TScl dn = D.norm( );
     stop  = ( dn < this->m_epsilon );
     stop |= ( std::isnan( dn ) || std::isinf( dn ) );
     stop |= ( ++i >= this->m_max_iter );
     stop |= this->m_Debug( &model, dn, i, &( this->m_CostFromCompleteData ) );
     } // end while
  */
}

#endif // __ivqML__Optimizer__ADAM__hxx__

// eof - $RCSfile$



/* TODO
   if( !this->has_model( ) )
   return;

   // Some values...
   static const TScl _1 = TScl( 1 );
   static const TScl _2 = TScl( 2 );
   static const TScl _10 = TScl( 10 );
   TScl e = std::pow( _10, std::log10( this->m_alpha ) * _2 );
   TScl b1t = this->m_beta1;
   TScl b2t = this->m_beta2;
   TScl b1 = _1 - this->m_beta1;
   TScl b2 = _1 - this->m_beta2;

   // Initialize optimizer
   TNat p = this->m_Model->number_of_parameters( );
   TRowMap mp = this->m_Model->row( p );
   bool stop = false;
   TNat i = 0;

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
   TScl J = this->m_Costs[ 0 ]( G.data( ) );
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
   TScl gn = G.norm( );
   stop  = ( gn < this->m_epsilon );
   stop |= ( std::isnan( gn ) || std::isinf( gn ) );
   stop |= ( ++i >= this->m_max_iterations );

   // Process debug information
   stop |=
   this->m_Debugger(
   J, gn, this->m_Model, i,
   stop || i == 1 || i % this->m_debug_iterations == 0
   );
*/

#endif // __ivqML__Optimizer__ADAM__hxx__

// eof - $RCSfile$
