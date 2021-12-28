// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Optimizer__Adam__hxx__
#define __PUJ__Optimizer__Adam__hxx__

// -------------------------------------------------------------------------
template< class _TModel >
PUJ::Optimizer::Adam< _TModel >::
Adam( )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::Adam< _TModel >::
AddArguments( boost::program_options::options_description* o )
{
  this->Superclass::AddArguments( o );
  o->add_options( )
    (
      "alpha",
      boost::program_options::value< TScalar >( &this->m_Alpha )->
      default_value( this->m_Alpha ),
      "learning rate"
      )
    (
      "beta1",
      boost::program_options::value< TScalar >( &this->m_Beta1 )->
      default_value( this->m_Beta1 ),
      "first decay coefficient"
      )
    (
      "beta2",
      boost::program_options::value< TScalar >( &this->m_Beta2 )->
      default_value( this->m_Beta2 ),
      "second decay coefficient"
      )
    ;
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::Adam< _TModel >::
Fit( )
{
  static const TScalar maxJ = std::numeric_limits< TScalar >::max( );
  static const TScalar _1 = TScalar( 1 );
  static const TScalar _2 = TScalar( 2 );
  TScalar J = maxJ, Jn, dJ;
  TScalar b1t = this->m_Beta1;
  TScalar b2t = this->m_Beta2;
  TRow g;
  TRow m = TRow::Zero( this->m_Cost->GetParameters( ).cols( ) );
  TRow v = TRow::Zero( this->m_Cost->GetParameters( ).cols( ) );
  bool stop = false;
  this->m_Iterations = 0;
  do
  {
    this->m_Iterations++;

    // Gradients
    for( unsigned int b = 0; b < this->m_Cost->GetNumberOfBatches( ); ++b )
    {
      Jn = this->m_Cost->operator()( b, &g );
      this->_Regularize( Jn, g );

      m = ( m * this->m_Beta1 ) + ( g * ( _1 - this->m_Beta1 ) );
      v =
        ( v * this->m_Beta2 ) +
        TRow( g.array( ).pow( _2 ) * ( _1 - this->m_Beta2 ) );
      this->m_Cost->operator-=(
        (
          ( m / ( _1 - b1t ) ).array( ) /
          ( ( v / ( _1 - b2t ) ).array( ).sqrt( ) + this->m_Epsilon )
          ) *
        this->m_Alpha
        );
    } // end if

    // Next step
    b1t *= this->m_Beta1;
    b2t *= this->m_Beta2;

    // Update cost difference
    dJ = ( J != maxJ )? J - Jn: J;

    // Update stop condition
    stop  =
      /* TODO: ( dJ <= this->m_Epsilon ) | */
      ( this->m_MaximumNumberOfIterations <= this->m_Iterations ) |
      this->m_Debug(
        this->m_Iterations, J, dJ,
        this->m_Iterations % this->m_DebugIterations == 0
        );

    // Ok, finished an iteration
    J = Jn;
  } while( !stop );

  // Finish iteration
  this->m_Debug( this->m_Iterations, J, dJ, true );
}

#endif // __PUJ__Optimizer__Adam__hxx__

// eof - $RCSfile$
