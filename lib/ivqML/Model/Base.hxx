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
    *( this->m_T + i ) = _S( *( other.m_T + i ) );
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
    *( this->m_T + i ) = _S( *( other.m_T + i ) );
  return( *this );
}

// -------------------------------------------------------------------------
template< class _S >
template< class _D >
typename ivqML::Model::Base< _S >::
Self& ivqML::Model::Base< _S >::
operator+=( const Eigen::EigenBase< _D >& d )
{
  TMap( this->m_T, d.rows( ), d.cols( ) )
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
  TMap( this->m_T, d.rows( ), d.cols( ) )
    -=
    d.derived( ).template cast< _S >( );
  return( *this );
}


#endif // __ivqML__Model__Base__hxx__

// eof - $RCSfile$
