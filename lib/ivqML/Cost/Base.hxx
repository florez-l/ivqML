// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__Base__hxx__
#define __ivqML__Cost__Base__hxx__

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
ivqML::Cost::Base< _M, _X, _Y >::
Base( const _M& m, const TX& iX, const TY& iY )
  : m_M( &m ),
    m_X( &iX ),
    m_Y( &iY )
{
  this->m_G =
    std::shared_ptr< TScalar[ ] >(
      new TScalar[ m.number_of_parameters( ) ] { 0 }
      );
  this->m_Ym =
    std::shared_ptr< TScalar[ ] >(
      new TScalar[ iY.derived( ).size( ) ] { 0 }
      );
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
ivqML::Cost::Base< _M, _X, _Y >::
~Base( )
{
  this->m_G.reset( );
  this->m_Ym.reset( );
}

#endif // __ivqML__Cost__Base__hxx__

// eof - $RCSfile$
