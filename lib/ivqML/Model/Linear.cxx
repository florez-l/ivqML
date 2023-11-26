// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/Linear.h>

// -------------------------------------------------------------------------
template< class _S >
ivqML::Model::Linear< _S >::
Linear( const TNatural& n )
  : Superclass( TNatural( 0 ) )
{
  this->set_number_of_parameters( n + 1 );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Linear< _S >::
set_number_of_parameters( const TNatural& p )
{
  this->Superclass::set_number_of_parameters( p );
  new( &this->m_T ) TMap( this->m_Parameters.data( ), p, 1 );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::Linear< _S >::
TNatural ivqML::Model::Linear< _S >::
number_of_inputs( ) const
{
  return( this->number_of_parameters( ) - 1 );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Linear< _S >::
set_number_of_inputs( const TNatural& i )
{
  this->set_number_of_parameters( i + 1 );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::Linear< _S >::
TNatural ivqML::Model::Linear< _S >::
number_of_outputs( ) const
{
  return( 1 );
}

// -------------------------------------------------------------------------
/* TODO
template< class _S >
void ivqML::Model::Linear< _S >::
_synch( )
{
  if( this->m_P > 0 )
    new( &this->m_M ) TMap( this->m_T.get( ) + 1, this->m_P - 1, 1 );
  else
    new( &this->m_M ) TMap( nullptr, 0, 0 );
}
*/
// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::Linear< _S >::
TNatural ivqML::Model::Linear< _S >::
_cache_size( ) const
{
  return(
    TNatural(
      double( this->m_Cache.size( ) )
      /
      double( this->m_Parameters.size( ) + 1 )
    )
  );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Linear< _S >::
_resize_cache( const TNatural& s ) const
{
  if( s < this->_cache_size( ) )
    return;

  TNatural n = this->number_of_inputs( );
  if( n > 0 && s > 0 )
  {
    this->Superclass::_resize_cache( s * ( n + 2 ) );
    std::fill( this->m_Cache.data( ), this->m_Cache.data( ) + s, 1 );
    new( &this->m_Xi ) TMap( this->m_Cache.data( ), s, n + 1 );
    new( &this->m_X ) TMap( this->m_Cache.data( ) + s, s, n );
    new( &this->m_Y ) TMap( this->m_Cache.data( ) + ( s * ( n + 1 ) ), s, 1 );
  }
  else
  {
    this->Superclass::_resize_cache( 0 );
    this->m_Xi = TMap( nullptr, 0, 0 );
    this->m_X = TMap( nullptr, 0, 0 );
    this->m_Y = TMap( nullptr, 0, 0 );
  } // end if
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::Linear< _S >::
TMap& ivqML::Model::Linear< _S >::
_input_cache( ) const
{
  return( this->m_X );
}

// -------------------------------------------------------------------------
template< class _S >
const typename ivqML::Model::Linear< _S >::
TMap& ivqML::Model::Linear< _S >::
_output_cache( ) const
{
  return( this->m_Y );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::Linear< _S >::
_evaluate( const TNatural& m ) const
{
  auto xi = this->m_Xi.block( 0, 0, m, this->m_Xi.cols( ) );
  auto y = this->m_Y.block( 0, 0, m, this->m_Y.cols( ) );
  y = xi * this->m_T;
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Linear< float >;
template class ivqML_EXPORT ivqML::Model::Linear< double >;
template class ivqML_EXPORT ivqML::Model::Linear< long double >;

// eof - $RCSfile$
