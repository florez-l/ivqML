// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Base__hxx__
#define __ivqML__Optimizer__Base__hxx__

#include <cmath>
#include <cstdlib>

// -------------------------------------------------------------------------
template< class _TCost >
ivqML::Optimizer::Base< _TCost >::
Base( )
{
  this->unset_debug( );
  this->m_epsilon
    =
    std::pow(
      TScl( 10 ),
      std::log10( std::numeric_limits< TScl >::epsilon( ) ) * TScl( 0.5 )
      );
}

// -------------------------------------------------------------------------
template< class _TCost >
ivqML::Optimizer::Base< _TCost >::
~Base( )
{
  if( this->m_D != nullptr )
    std::free( this->m_D );
}

// -------------------------------------------------------------------------
template< class _TCost >
void ivqML::Optimizer::Base< _TCost >::
register_options( boost::program_options::options_description& opt )
{
  opt.add_options( )
    ivqML_Optimizer_OptionMacro( batch_size, "batch_size" )
    ivqML_Optimizer_OptionMacro( lambda, "lambda,l" )
    ivqML_Optimizer_OptionMacro( epsilon, "epsilon,e" )
    ivqML_Optimizer_OptionMacro( max_iter, "max_iterations,I" );
}

// -------------------------------------------------------------------------
template< class _TCost >
template< class _TInputX, class _TInputY >
void ivqML::Optimizer::Base< _TCost >::
set_data(
  const Eigen::EigenBase< _TInputX >& iX,
  const Eigen::EigenBase< _TInputY >& iY
  )
{
  this->m_M = iX.cols( );
  this->m_N = iX.rows( );
  this->m_P = iY.rows( );
  TNat b = ( this->m_batch_size > 0 )? this->m_batch_size: this->m_M;
  TNat s = this->m_M / b;
  TNat l = this->m_M % b;

  // Memory management
  if( this->m_D != nullptr )
    std::free( this->m_D );
  this->m_D =
    reinterpret_cast< TScl* >(
      std::calloc(
        ( this->m_N + this->m_P ) * this->m_M,
        sizeof( TScl )
        )
      );
  TScl* X = this->m_D;
  TScl* Y = this->m_D + ( this->m_N * this->m_M );

  // Cast and copy input data
  TMatMap( X, this->m_N, this->m_M ) = iX.derived( ).template cast< TScl >( );
  TMatMap( Y, this->m_P, this->m_M ) = iY.derived( ).template cast< TScl >( );

  // A cost function that process all data
  this->m_CostFromCompleteData.set_data(
    X, Y, this->m_M, this->m_N, this->m_P
    );

  this->m_Costs.clear( );
  this->m_Costs.resize( s + ( ( l > 0 )? 1: 0 ) );
  this->m_Costs.shrink_to_fit( );
  for( TNat i = 0; i < s; ++i )
  {
    this->m_Costs[ i ].set_data(
      X + ( this->m_N * b * i ),
      Y + ( this->m_P * b * i ),
      b, this->m_N, this->m_P
      );

  } // end for
  if( l > 0 )
    this->m_Costs.back( ).set_data(
      X + ( this->m_N * b * s ),
      Y + ( this->m_P * b * s ),
      l, this->m_N, this->m_P
      );
}

// -------------------------------------------------------------------------
template< class _TCost >
void ivqML::Optimizer::Base< _TCost >::
unset_debug( )
{
  this->m_Debug =
    []( const TModel* m, const TScl& n, const TNat& i, const TCost* c, bool f )
    ->
    bool
    {
      return( false );
    };
}

#endif // __ivqML__Optimizer__Base__hxx__

// eof - $RCSfile$
