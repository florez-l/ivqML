// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Base__hxx__
#define __ivqML__Optimizer__Base__hxx__

// -------------------------------------------------------------------------
template< class _C >
ivqML::Optimizer::Base< _C >::
Base( TModel& m, const TX& iX, const TY& iY )
  : m_M( &m ),
    m_X( &iX ),
    m_Y( &iY )
{
  this->m_P.add_options( )( "help,h", "help message" )
    ivqML_Optimizer_OptionMacro( lambda, "lambda,l" )
    ivqML_Optimizer_OptionMacro( max_iterations, "max_iterations,M" )
    ivqML_Optimizer_OptionMacro( debug_iterations, "debug_iterations,D" );

  // TODO: this->_configure_parameter( "regularization", "ridge" );
}

// -------------------------------------------------------------------------
template< class _C >
std::string ivqML::Optimizer::Base< _C >::
parse_options( int argc, char** argv )
{
  boost::program_options::variables_map m;
  boost::program_options::store(
    boost::program_options::parse_command_line( argc, argv, this->m_P ), m
    );
  boost::program_options::notify( m );
  if( m.count( "help" ) )
  {
    std::stringstream r;
    r << this->m_P;
    return( r.str( ) );
  }
  else
    return( "" );
}

// -------------------------------------------------------------------------
template< class _C >
void ivqML::Optimizer::Base< _C >::
set_debug( TDebug d )
{
  this->m_D = d;
}

#endif // __ivqML__Optimizer__Base__hxx__

// eof - $RCSfile$
