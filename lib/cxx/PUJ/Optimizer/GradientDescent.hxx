// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Optimizer__GradientDescent__hxx__
#define __PUJ__Optimizer__GradientDescent__hxx__

// -------------------------------------------------------------------------
template< class _TModel >
PUJ::Optimizer::GradientDescent< _TModel >::
GradientDescent( )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
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
    ;
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
Fit( )
{
  static const TScalar maxJ = std::numeric_limits< TScalar >::max( );
  TScalar J = maxJ, Jn, dJ;
  TRow g;
  bool stop = false;
  this->m_Iterations = 0;
  do
  {
    // Next iteration
    for( unsigned int b = 0; b < this->m_Cost->GetNumberOfBatches( ); ++b )
    {
      Jn = this->m_Cost->operator()( b, &g );
      this->_Regularize( Jn, g );
      this->m_Cost->operator-=( g * this->m_Alpha );
    } // end if

    // Update cost difference
    dJ = ( J != maxJ )? J - Jn: J;

    // Update stop condition
    stop  =
      ( dJ <= this->m_Epsilon ) |
      ( this->m_MaximumNumberOfIterations <= this->m_Iterations ) |
      this->m_Debug(
        this->m_Iterations, J, dJ,
        this->m_Iterations % this->m_DebugIterations == 0
        );

    // Ok, finished an iteration
    J = Jn;
    this->m_Iterations++;
  } while( !stop );

  // Finish iteration
  this->m_Debug( this->m_Iterations, J, dJ, true );
}

#endif // __PUJ__Optimizer__GradientDescent__hxx__

// eof - $RCSfile$
