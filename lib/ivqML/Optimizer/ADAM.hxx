// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__ADAM__hxx__
#define __ivqML__Optimizer__ADAM__hxx__

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
ivqML::Optimizer::ADAM< _M, _X, _Y >::
ADAM( )
  : Superclass( )
{
  this->m_P.add_options( )
    ivqML_Optimizer_OptionMacro( alpha, "alpha,a" )
    ivqML_Optimizer_OptionMacro( beta1, "beta1,b" )
    ivqML_Optimizer_OptionMacro( beta2, "beta2,c" );
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
void ivqML::Optimizer::ADAM< _M, _X, _Y >::
fit( )
{
  static const TScalar _1 = TScalar( 1 );
  static const TScalar _2 = TScalar( 2 );
  static const TScalar _10 = TScalar( 10 );
  TScalar e = std::pow( _10, std::log10( this->m_alpha ) * _2 );
  TScalar b1t = this->m_beta1;
  TScalar b2t = this->m_beta2;
  TScalar b1 = _1 - this->m_beta1;
  TScalar b2 = _1 - this->m_beta2;

  // Prepare loop
  TMatrix G( 1, this->m_M->number_of_parameters( ) );
  TNatural B = this->m_Sizes.size( );
  TMatrix M = TMatrix::Zero( G.rows( ), G.cols( ) );
  TMatrix V = TMatrix::Zero( G.rows( ), G.cols( ) );
  TMatrix M2 = M;
  TMatrix V2 = V;

  // Main loop
  bool stop = false, debug_stop = false;
  TNatural i = 0;
  while( !stop )
  {
    // Compute gradient
    M *= this->m_beta1;
    V *= this->m_beta2;
    TNatural j = 0;
    for( const TNatural& s: this->m_Sizes )
    {
      this->m_M->cost(
        G,
        this->m_X->derived( ).block( j, 0, s, this->m_X->cols( ) ),
        this->m_Y->derived( ).block( j, 0, s, this->m_Y->cols( ) )
        );
      j += s;
      M2 = ( M + ( G * b1 ) ) / ( _1 - b1t );
      V2 =
        ( ( V.array( ) + ( G.array( ).pow( 2 ) * b2 ) ) / ( _1 - b2t ) )
        .sqrt( );
      *( this->m_M ) -=
        ( ( M2.array( ) / ( V2.array( ) + e ) ) * this->m_alpha ).matrix( );
    } // end for
    M = M2;
    V = V2;

    // Prepare next step
    debug_stop =
      this->m_D(
        0, G.norm( ), this->m_M, i,
        ( i % this->m_debug_iterations == 0 )
        );
    i++;
    b1t *= this->m_beta1;
    b2t *= this->m_beta2;
    stop =
      ( G.norm( ) <= e )
      ||
      ( this->m_max_iterations == i )
      ||
      debug_stop;
  } // end while
  this->m_D( 0, G.norm( ), this->m_M, i, true );
}

#endif // __ivqML__Optimizer__ADAM__hxx__

// eof - $RCSfile$
