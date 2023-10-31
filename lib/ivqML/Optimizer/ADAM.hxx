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
  this->configure_parameter( "alpha", TScalar( 1e-3 ) );
  this->configure_parameter( "beta1", TScalar( 0.9 ) );
  this->configure_parameter( "beta2", TScalar( 0.999 ) );
}

// -------------------------------------------------------------------------
template< class _C >
void ivqML::Optimizer::ADAM< _C >::
fit( )
{
  static const TScalar _1 = TScalar( 1 );

  TScalar a, b1, b2;
  TNatural I, Di;
  this->parameter( a, "alpha" );
  this->parameter( b1, "beta1" );
  this->parameter( b2, "beta2" );
  this->parameter( I, "max_iterations" );
  this->parameter( Di, "debug_iterations" );

  TScalar e = std::pow( TScalar( 10 ), std::log10( a ) * TScalar( 2 ) );
  TScalar b1t = b1;
  TScalar b2t = b2;

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
    M = ( ( M * b1 ) + ( G * ( _1 - b1 ) ) ) / ( _1 - b1t );
    V =
      (
        ( ( V.array( ) * b2 ) + ( G.array( ).pow( 2 ) * ( _1 - b2 ) ) )
        /
        ( _1 - b2t )
      )
      .sqrt( );
    *( this->m_M ) -= ( ( M.array( ) / ( V.array( ) + e ) ) * a ).matrix( );

    if( i % Di == 0 )
      debug_stop = this->m_D( J.first, G.norm( ), this->m_M, i );
    i++;
    b1t *= b1;
    b2t *= b2;
    stop = ( G.norm( ) <= e ) || ( I == i ) || debug_stop;
    if( !stop )
      J = cost( );
  } // end while
  this->m_D( J.first, G.norm( ), this->m_M, i );
}

#endif // __ivqML__Optimizer__ADAM__hxx__

// eof - $RCSfile$
