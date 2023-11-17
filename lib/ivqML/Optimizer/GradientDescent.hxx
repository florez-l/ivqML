// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__GradientDescent__hxx__
#define __ivqML__Optimizer__GradientDescent__hxx__

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
ivqML::Optimizer::GradientDescent< _M, _X, _Y >::
GradientDescent( )
  : Superclass( )
{
  this->m_P.add_options( )
    ivqML_Optimizer_OptionMacro( alpha, "alpha,a" );
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
void ivqML::Optimizer::GradientDescent< _M, _X, _Y >::
fit( )
{
  static const TScalar _2 = TScalar( 2 );
  static const TScalar _10 = TScalar( 10 );
  TScalar e = std::pow( _10, std::log10( this->m_alpha ) * _2 );

  // Prepare loop
  TScalar J = std::numeric_limits< TScalar >::max( );
  TMatrix G( 1, this->m_M->number_of_parameters( ) );
  TNatural B = this->m_Sizes.size( );

  // Main loop
  bool stop = false, debug_stop = false;
  TNatural i = 0;
  while( !stop )
  {
    TNatural j = 0;
    for( const TNatural& s: this->m_Sizes )
    {
      J =
        this->m_M->cost(
          G,
          this->m_X->derived( ).block( j, 0, s, this->m_X->cols( ) ),
          this->m_Y->derived( ).block( j, 0, s, this->m_Y->cols( ) )
          );
      j += s;
      *( this->m_M ) -= G * this->m_alpha;
    } // end for

    debug_stop =
      this->m_D(
        J, G.norm( ), this->m_M, i,
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
  this->m_D( J, G.norm( ), this->m_M, i, true );
}

#endif // __ivqML__Optimizer__GradientDescent__hxx__

// eof - $RCSfile$
