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
  this->configure_parameter( "lambda", TScalar( 0 ) );
  this->configure_parameter( "regularization", "ridge" );
  this->configure_parameter(
    "max_iterations", std::numeric_limits< TNatural >::max( )
    );
  this->configure_parameter( "debug_iterations", TNatural( 100 ) );
}

// -------------------------------------------------------------------------
template< class _C >
template< class _V >
bool ivqML::Optimizer::Base< _C >::
parameter( _V& v, const std::string& n ) const
{
  auto p = this->m_P.find( n );
  if( p != this->m_P.end( ) )
  {
    std::istringstream s( p->second );
    s >> v;
    return( true );
  }
  else
    return( false );
}

// -------------------------------------------------------------------------
template< class _C >
template< class _V >
void ivqML::Optimizer::Base< _C >::
configure_parameter( const std::string& n, const _V& v )
{
  auto p = this->m_P.insert( std::make_pair( n, "" ) );
  std::stringstream s;
  s << v;
  p.first->second = s.str( );
}

// -------------------------------------------------------------------------
template< class _C >
template< class _V >
void ivqML::Optimizer::Base< _C >::
set_parameter( const std::string& n, const _V& v )
{
  auto p = this->m_P.find( n );
  if( p != this->m_P.end( ) )
  {
    std::stringstream s;
    s << v;
    p->second = s.str( );
  } // end if
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
