// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Base__hxx__
#define __ivqML__Model__Base__hxx__

// -------------------------------------------------------------------------
/* TODO
   template< class _TScalar >
   template< class _TOther >
   void ivqML::Model::Base< _TScalar >::
   shallow_copy( const _TOther& other )
   {
   this->set_number_of_parameters( other.m_Size );
   this->random_fill( );
   }

   // -------------------------------------------------------------------------
   template< class _TScalar >
   template< class _TOther >
   void ivqML::Model::Base< _TScalar >::
   deep_copy( const _TOther& other )
   {
   this->shallow_copy( other );
   for( TNatural i = 0; i < this->m_Size; ++i )
   *( this->m_Parameters + i ) = TScalar( *( other.m_Parameters + i ) );
   }
*/

#endif // __ivqML__Model__Base__hxx__

// eof - $RCSfile$
