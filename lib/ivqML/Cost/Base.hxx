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
  this->m_G = new TScalar[ m.number_of_parameters( ) ];
  this->m_Ym = new TScalar[ iY.derived( ).size( ) ];
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
ivqML::Cost::Base< _M, _X, _Y >::
~Base( )
{
  if( this->m_G != nullptr )
    delete this->m_G;
  if( this->m_Ym != nullptr )
    delete this->m_Ym;
}

#endif // __ivqML__Cost__Base__hxx__

// eof - $RCSfile$
