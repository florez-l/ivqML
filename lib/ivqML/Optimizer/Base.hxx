// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Base__hxx__
#define __ivqML__Optimizer__Base__hxx__

#include <algorithm>

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
ivqML::Optimizer::Base< _M, _X, _Y >::
Base( )
{
  this->m_P.add_options( )( "help,h", "help message" )
    ivqML_Optimizer_OptionMacro( lambda, "lambda,l" )
    ivqML_Optimizer_OptionMacro( batch_size, "batch_size" )
    ivqML_Optimizer_OptionMacro( max_iterations, "max_iterations,M" )
    ivqML_Optimizer_OptionMacro( debug_iterations, "debug_iterations,D" );

  // TODO: this->_configure_parameter( "regularization", "ridge" );
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
std::string ivqML::Optimizer::Base< _M, _X, _Y >::
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
template< class _M, class _X, class _Y >
void ivqML::Optimizer::Base< _M, _X, _Y >::
init( TModel& m, const TX& iX, const TY& iY )
{
  this->m_M = &m;
  this->m_X = &iX;
  this->m_Y = &iY;

  TNatural M = iX.rows( );
  TNatural B = this->m_batch_size;
  B = ( B == 0 || B > M )? M: B;
  TNatural N = std::ceil( double( M ) / double( B ) );

  this->m_Sizes.resize( N );
  this->m_Sizes.shrink_to_fit( );

  std::fill( this->m_Sizes.begin( ), this->m_Sizes.end( ), B );
  if( ( M % B ) > 0 )
    this->m_Sizes.back( ) = M % B;
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
void ivqML::Optimizer::Base< _M, _X, _Y >::
set_debug( TDebug d )
{
  this->m_D = d;
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
std::vector< std::pair< typename ivqML::Optimizer::Base< _M, _X, _Y >::TMap, typename ivqML::Optimizer::Base< _M, _X, _Y >::TMap > > ivqML::Optimizer::Base< _M, _X, _Y >::
_batches( )
{
  std::vector< std::pair< TMap, TMap > > b;

  // Prepare buffer to keep all used data
  TNatural Xs = this->m_X->size( );
  TNatural Xr = this->m_X->rows( );
  TNatural Xc = this->m_X->cols( );
  TNatural Ys = this->m_Y->size( );
  TNatural Yr = this->m_Y->rows( );
  TNatural Yc = this->m_Y->cols( );
  this->m_Buffer.resize( Xs + Ys );
  this->m_Buffer.shrink_to_fit( );

  // Cast and copy all data
  TScalar* buf = this->m_Buffer.data( );
  TMap( buf, Xr, Xc ) = this->m_X->derived( ).template cast< TScalar >( );
  TMap( buf + Xs, Yr, Yc ) = this->m_Y->derived( ).template cast< TScalar >( );

  TNatural i = 0, j = Xs;
  for( const TNatural& s: this->m_Sizes )
  {
    b.push_back(
      std::make_pair(
        TMap( buf + i, s, Xc ),
        TMap( buf + j, s, Yc )
        )
      );
    i += b.back( ).first.size( );
    j += b.back( ).second.size( );
  } // end if
  b.shrink_to_fit( );

  return( b );
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
void ivqML::Optimizer::Base< _M, _X, _Y >::
_clear_batches( )
{
  this->m_Buffer.clear( );
}

#endif // __ivqML__Optimizer__Base__hxx__

// eof - $RCSfile$
