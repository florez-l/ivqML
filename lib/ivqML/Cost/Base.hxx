// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__Base__hxx__
#define __ivqML__Cost__Base__hxx__

#include <cmath>

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
ivqML::Cost::Base< _M, _X, _Y >::
Base( const _M& m, const TX& iX, const TY& iY )
  : m_M( &m ),
    m_X( &iX ),
    m_Y( &iY )
{
  this->m_G = TMatrix::Zero( 1, m.number_of_parameters( ) );
  this->m_Z = TMatrix::Zero( iY.rows( ), iY.cols( ) );
  this->set_batch_size( 0 );
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
typename ivqML::Cost::Base< _M, _X, _Y >::
TNatural ivqML::Cost::Base< _M, _X, _Y >::
number_of_batches( ) const
{
  return( this->m_B.size( ) );
}

// -------------------------------------------------------------------------
template< class _M, class _X, class _Y >
void ivqML::Cost::Base< _M, _X, _Y >::
set_batch_size( const TNatural& bs )
{
  this->m_B.clear( );
  TNatural m = this->m_X->rows( );
  TNatural nb = 1;
  if( bs > 0 )
    nb = TNatural( std::ceil( double( m ) / double( bs ) ) );
  if( nb == 0 )
    nb = 1;

  if( nb > 1 )
  {
    for( TNatural b = 0; b < nb; ++b )
    {
      TNatural i = b * bs;
      TNatural j = i + bs;
      this->m_B.push_back( std::make_pair( i, ( ( j > m )? m: j ) - i ) );
    } // end for
  }
  else
    this->m_B.push_back( std::make_pair( 0, m ) );
}

#endif // __ivqML__Cost__Base__hxx__

// eof - $RCSfile$
