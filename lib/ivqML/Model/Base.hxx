// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Base__hxx__
#define __ivqML__Model__Base__hxx__

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TG >
typename ivqML::Model::Base< _TReal, _TNatural >::
Self& ivqML::Model::Base< _TReal, _TNatural >::
operator+=( const Eigen::EigenBase< _TG >& G )
{
  TMatrixMap( this->m_P, G.rows( ), G.cols( ) )
    +=
    G.derived( ).template cast< TReal >( );
  return( *this );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
template< class _TG >
typename ivqML::Model::Base< _TReal, _TNatural >::
Self& ivqML::Model::Base< _TReal, _TNatural >::
operator-=( const Eigen::EigenBase< _TG >& G )
{
  TMatrixMap( this->m_P, G.rows( ), G.cols( ) )
    -=
    G.derived( ).template cast< TReal >( );
  return( *this );
}

#endif // __ivqML__Model__Base__hxx__

// eof - $RCSfile$
