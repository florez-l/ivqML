// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Optimizer__GradientDescent__hxx__
#define __PUJ__Optimizer__GradientDescent__hxx__

#include <boost/program_options.hpp>

// -------------------------------------------------------------------------
template< class _TModel >
PUJ::Optimizer::GradientDescent< _TModel >::
GradientDescent( )
  : m_Cost( nullptr )
{
  this->m_Debug =
    []( unsigned long long, TScalar, TScalar, bool ) -> bool
    { return( false ); };
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
SetCost( TCost* cost )
{
    std::cout << "Set " << cost << std::endl;
    this->m_Cost = cost;
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
AddArguments( boost::program_options::options_description* o )
{
  o->add_options( )
    (
      "alpha",
      boost::program_options::value< TScalar >( &this->m_Alpha )->
      default_value( this->m_Alpha ),
      "learning rate"
    )
    (
      "lambda",
      boost::program_options::value< TScalar >( &this->m_Lambda )->
      default_value( this->m_Lambda ),
      "regularization"
    )
    (
      "max_iter",
      boost::program_options::value< unsigned long long >(
        &this->m_MaximumNumberOfIterations
        )->
      default_value( this->m_MaximumNumberOfIterations ),
      "maximum iterations"
    )
    (
      "deb_iter",
      boost::program_options::value< unsigned long long >(
        &this->m_DebugIterations
        )->
      default_value( this->m_DebugIterations ),
      "iterations for debug"
    )
    ;
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
SetRegularizationTypeToRidge( )
{
  this->SetRegularizationType( Self::RidgeRegType );
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
SetRegularizationTypeToLASSO( )
{
  this->SetRegularizationType( Self::LASSORegType );
}

// -------------------------------------------------------------------------
template< class _TModel >
const unsigned long long& PUJ::Optimizer::GradientDescent< _TModel >::
GetIterations( ) const
{
  return( this->m_Iterations );
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
SetDebug( TDebug d )
{
  this->m_Debug = d;
}

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
Fit( )
{
    std::cout << "Start " << this->m_Cost << std::endl;
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

// -------------------------------------------------------------------------
template< class _TModel >
void PUJ::Optimizer::GradientDescent< _TModel >::
_Regularize( TScalar& J, TRow& g )
{
  if( this->m_Lambda != TScalar( 0 ) )
  {
    const TRow& t = this->m_Cost->GetParameters( );
    if( this->m_RegularizationType == Self::RidgeRegType )
    {
      J += t.squaredNorm( ) * this->m_Lambda;
      g += t * TScalar( 2 ) * this->m_Lambda;
    }
    else if( this->m_RegularizationType == Self::LASSORegType )
    {
      J += t.array( ).abs( ).sum( ) * this->m_Lambda;
      // TODO:
    } // end if
  } // end if
}

#endif // __PUJ__Optimizer__GradientDescent__hxx__

// eof - $RCSfile$
