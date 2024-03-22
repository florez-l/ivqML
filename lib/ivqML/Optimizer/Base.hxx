// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Base__hxx__
#define __ivqML__Optimizer__Base__hxx__

#include <cmath>

// -------------------------------------------------------------------------
template< class _TCost >
ivqML::Optimizer::Base< _TCost >::
Base( )
{
  // TODO: this->unset_debugger( );
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
  this->m_X = iX.derived( ).template cast< TScl >( );
  this->m_Y = iY.derived( ).template cast< TScl >( );

#error ACA VOY

  TNat m = this->m_X.cols( );
  TNat b = ( this->m_batch_size > 0 )? this->m_batch_size: m;
  TNat n = m / b;
  TNat l = m % b;

  this->m_Costs.clear( );
  this->m_Costs.resize( n + ( ( l > 0 )? 1: 0 ) );
  this->m_Costs.shrink_to_fit( );
  for( TNat i = 0; i < n; ++i )
  {
    this->m_Costs[ i ].set_data(
      this->m_X.data( ), this->m_Y.data( ),
      m, this->m_X.rows( ), this->m_Y.rows( )
      );

    /* TODO
       this->m_Costs[ i ].set_data(
       X.block( 0, b * i, X.rows( ), b ), Y.block( 0, b * i, Y.rows( ), b )
       );
    */
  } // end for
  /* TODO
     if( l > 0 )
     this->m_Costs.back( ).set_data(
     X.block( 0, b * n, X.rows( ), l ), Y.block( 0, b * n, Y.rows( ), l )
     );
  */
}

#endif // __ivqML__Optimizer__Base__hxx__

// eof - $RCSfile$
