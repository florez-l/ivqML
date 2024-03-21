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
  this->unset_debugger( );
  this->m_epsilon
    =
    std::pow(
      TScalar( 10 ),
      std::log10( std::numeric_limits< TScalar >::epsilon( ) )
      *
      TScalar( 0.5 )
      );

  this->m_Options.add_options( )( "help,h", "help message" )
    ivqML_Optimizer_OptionMacro( lambda, "lambda,l" )
    ivqML_Optimizer_OptionMacro( epsilon, "epsilon,e" )
    ivqML_Optimizer_OptionMacro( batch_size, "batch_size" )
    ivqML_Optimizer_OptionMacro( max_iterations, "max_iterations,M" )
    ivqML_Optimizer_OptionMacro( debug_iterations, "debug_iterations,D" );
}

// -------------------------------------------------------------------------
template< class _TCost >
ivqML::Optimizer::Base< _TCost >::
~Base( )
{
  if( this->m_Model != nullptr && this->m_ManagedModel )
    delete this->m_Model;
}

// -------------------------------------------------------------------------
template< class _TCost >
std::string ivqML::Optimizer::Base< _TCost >::
parse_arguments( int c, char** v )
{
  boost::program_options::variables_map m;
  boost::program_options::store(
    boost::program_options::command_line_parser( c, v )
    .options( this->m_Options ).allow_unregistered( ).run( ),
    m
    );
  boost::program_options::notify( m );
  if( m.count( "help" ) )
  {
    std::stringstream r;
    r << this->m_Options;
    return( r.str( ) );
  }
  else
    return( "" );
}

// -------------------------------------------------------------------------
template< class _TCost >
bool ivqML::Optimizer::Base< _TCost >::
has_model( ) const
{
  return( this->m_Model != nullptr );
}

// -------------------------------------------------------------------------
template< class _TCost >
typename ivqML::Optimizer::Base< _TCost >::
TModel& ivqML::Optimizer::Base< _TCost >::
model( )
{
  return( *( this->m_Model ) );
}

// -------------------------------------------------------------------------
template< class _TCost >
const typename ivqML::Optimizer::Base< _TCost >::
TModel& ivqML::Optimizer::Base< _TCost >::
model( ) const
{
  return( *( this->m_Model ) );
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
  if( this->m_Model == nullptr )
  {
    if( this->m_Model != nullptr && this->m_ManagedModel )
      delete this->m_Model;
    this->m_Model = new TModel( );
    this->m_Model->set_number_of_inputs( iX.rows( ) );
    this->m_Model->random_fill( );
    this->m_ManagedModel = true;
  } // end if

  this->m_Costs.push_back( TCost( *( this->m_Model ) ) );
  this->m_Costs.back( ).set_data( iX, iY );
  this->m_Costs.shrink_to_fit( );
}

// -------------------------------------------------------------------------
template< class _TCost >
void ivqML::Optimizer::Base< _TCost >::
set_debugger( TDebugger d )
{
  this->m_Debugger = d;
}

// -------------------------------------------------------------------------
template< class _TCost >
void ivqML::Optimizer::Base< _TCost >::
unset_debugger( )
{
  this->m_Debugger =
    []( const TScalar&, const TScalar&, const TModel*, const TNatural&, bool )
    -> bool
    {
      return( false );
    };
}

#endif // __ivqML__Optimizer__Base__hxx__

// eof - $RCSfile$
