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
    ivqML_Optimizer_OptionMacro( batch_size, "batch_size" )
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

// -------------------------------------------------------------------------
template< class _C >
void ivqML::Optimizer::Base< _C >::
_batches( std::vector< TBX >& X, std::vector< TBY >& Y )
{
  auto all_X = this->m_X->derived( );
  auto all_Y = this->m_Y->derived( );
  X.clear( );
  Y.clear( );
  TNatural m = X.rows( );
  TNatural nb = 1;
  if( this->m_batch_size > 0 )
    nb = TNatural( std::ceil( double( m ) / double( this->m_batch_size ) ) );
  if( nb == 0 )
    nb = 1;

  if( nb > 1 )
  {
    for( TNatural b = 0; b < nb; ++b )
    {
      X.push_back( all_X.block( ???, 0, ???, all_X.cols( ) ) );
      Y.push_back( all_Y.block( ???, 0, ???, all_Y.cols( ) ) );
    } // end for
  }
  else
  {
    X.push_back( all_X.block( 0, 0, all_X.rows( ), all_X.cols( ) ) );
    Y.push_back( all_Y.block( 0, 0, all_Y.rows( ), all_Y.cols( ) ) );
  } // end if
}

#endif // __ivqML__Optimizer__Base__hxx__

// eof - $RCSfile$
