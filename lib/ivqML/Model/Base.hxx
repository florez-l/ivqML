// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Base__hxx__
#define __ivqML__Model__Base__hxx__

// -------------------------------------------------------------------------
template< class _S >
template< class _O >
ivqML::Model::Base< _S >::
Base( const _O& other )
{
  this->set_number_of_parameters( other.m_P );
  for( TNatural i = 0; i < this->m_P; ++i )
    *( this->m_T.get( ) + i ) = _S( *( other.m_T.get( ) + i ) );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _O >
typename ivqML::Model::Base< _S >::
Self& ivqML::Model::Base< _S >::
operator=( const _O& other )
{
  this->set_number_of_parameters( other.m_P );
  for( TNatural i = 0; i < this->m_P; ++i )
    *( this->m_T.get( ) + i ) = _S( *( other.m_T.get( ) + i ) );
  return( *this );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _D >
typename ivqML::Model::Base< _S >::
Self& ivqML::Model::Base< _S >::
operator+=( const Eigen::EigenBase< _D >& d )
{
  TMap( this->m_T.get( ), d.rows( ), d.cols( ) )
    +=
    d.derived( ).template cast< _S >( );
  return( *this );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _D >
typename ivqML::Model::Base< _S >::
Self& ivqML::Model::Base< _S >::
operator-=( const Eigen::EigenBase< _D >& d )
{
  TMap( this->m_T.get( ), d.rows( ), d.cols( ) )
    -=
    d.derived( ).template cast< _S >( );
  return( *this );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _X >
auto ivqML::Model::Base< _S >::
evaluate( const Eigen::EigenBase< _X >& iX ) const
{
  TNatural m = iX.rows( );
  TNatural o = this->number_of_outputs( );
  TNatural i = this->number_of_inputs( );

  this->resize_cache( m );
  this->_input_cache( ).block( 0, 0, m, i )
    =
    iX.derived( ).template cast< TScalar >( );
  this->_evaluate( m );
  return( this->_output_cache( ).block( 0, 0, m, o ) );
}

#endif // __ivqML__Model__Base__hxx__

// eof - $RCSfile$
