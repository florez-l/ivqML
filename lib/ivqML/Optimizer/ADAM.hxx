// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__ADAM__hxx__
#define __ivqML__Optimizer__ADAM__hxx__

// -------------------------------------------------------------------------
template< class _C >
ivqML::Optimizer::ADAM< _C >::
ADAM( TModel& m, const TX& iX, const TY& iY )
  : Superclass( m, iX, iY )
{
  this->m_P.add_options( )
    ivqML_Optimizer_OptionMacro( alpha, "alpha,a" )
    ivqML_Optimizer_OptionMacro( beta1, "beta1,b" )
    ivqML_Optimizer_OptionMacro( beta2, "beta2,c" );
}

// -------------------------------------------------------------------------
template< class _C >
void ivqML::Optimizer::ADAM< _C >::
fit( )
{
  static const TScalar _1 = TScalar( 1 );
  TScalar e =
    std::pow( TScalar( 10 ), std::log10( this->m_alpha ) * TScalar( 2 ) );
  TScalar b1t = this->m_beta1;
  TScalar b2t = this->m_beta2;
  TScalar b1 = _1 - this->m_beta1;
  TScalar b2 = _1 - this->m_beta2;

  // Cost function
  _C cost( *( this->m_M ), *( this->m_X ), *( this->m_Y ) );
  auto J = cost( );
  auto G = TConstMap( J.second, 1, this->m_M->number_of_parameters( ) );
  TMatrix M = TMatrix::Zero( G.rows( ), G.cols( ) );
  TMatrix V = TMatrix::Zero( G.rows( ), G.cols( ) );

  // Main loop
  bool stop = false, debug_stop = false;
  TNatural i = 0;
  while( !stop )
  {
    M = ( ( M * this->m_beta1 ) + ( G * b1 ) ) / ( _1 - b1t );
    V =
      (
        ( ( V.array( ) * this->m_beta2 ) + ( G.array( ).pow( 2 ) * b2 ) )
        /
        ( _1 - b2t )
      )
      .sqrt( );
    *( this->m_M ) -=
      ( ( M.array( ) / ( V.array( ) + e ) ) * this->m_alpha ).matrix( );

    if( i % this->m_debug_iterations == 0 )
      debug_stop = this->m_D( J.first, G.norm( ), this->m_M, i );
    i++;
    b1t *= this->m_beta1;
    b2t *= this->m_beta2;
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

#endif // __ivqML__Optimizer__ADAM__hxx__

// eof - $RCSfile$
