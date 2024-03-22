// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__Base__hxx__
#define __ivqML__Cost__Base__hxx__

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Cost::Base< _TModel >::
Base( )
{
}

// -------------------------------------------------------------------------
template< class _TModel >
ivqML::Cost::Base< _TModel >::
~Base( )
{
  /* TODO
     if( this->m_B != nullptr )
     delete this->m_B;
  */
}

// -------------------------------------------------------------------------
/* TODO
   template< class _TModel >
   template< class _TInputX, class _TInputY >
   void ivqML::Cost::Base< _TModel >::
   set_data(
   const Eigen::EigenBase< _TInputX >& iX,
   const Eigen::EigenBase< _TInputY >& iY
   )
   {
   this->m_X = iX.derived( ).template cast< TScl >( );
   this->m_Y = iY.derived( ).template cast< TScl >( );
   }
*/
 
// -------------------------------------------------------------------------
template< class _TModel >
void ivqML::Cost::Base< _TModel >::
set_data( TScl* X, TScl* Y, const TNat& m, const TNat& n, const TNat& p )
{
  new ( &this->m_X ) TMatMap( X, n, m );
  new ( &this->m_Y ) TMatMap( Y, p, m );
}

// -------------------------------------------------------------------------
/* TODO
   template< class _TModel >
   void ivqML::Cost::Base< _TModel >::
   set_model( TModel* m )
   {
   this->m_M = m;
   }
*/

#endif // __ivqML__Cost__Base__hxx__

// eof - $RCSfile$
