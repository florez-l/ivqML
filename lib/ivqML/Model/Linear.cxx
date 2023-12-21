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
  this->_map( this->m_T, p - 1, 1, 1 );
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
   cost( TMatrix& G, const TMap& X, const TMap& Y, TScalar* J ) const
   {
   TMatrix D = this->evaluate( X ).array( ) - Y.array( );

   G( 0, 0 ) = TScalar( 2 ) * D.mean( );
   G.block( 0, 1, 1, G.cols( ) - 1 )
   =
   ( ( D.transpose( ) * X ) * ( TScalar( 2 ) / TScalar( X.rows( ) ) ) );
   if( J != nullptr )
   *J = D.array( ).pow( 2 ).mean( );
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
   typename ivqML::Model::Linear< _S >::
   TMap& ivqML::Model::Linear< _S >::
   _output_cache( ) const
   {
   return( this->m_Y );
   }
*/

// -------------------------------------------------------------------------
/* TODO
   template< class _S >
   void ivqML::Model::Linear< _S >::
   _evaluate( const TNatural& m ) const
   {
   auto Xi = this->m_Xi.block( 0, 0, m, this->m_Xi.cols( ) );
   auto Y = this->m_Y.block( 0, 0, m, this->m_Y.cols( ) );
   Y = Xi * this->m_T;
   }
*/

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::Linear< float >;
template class ivqML_EXPORT ivqML::Model::Linear< double >;
template class ivqML_EXPORT ivqML::Model::Linear< long double >;

// eof - $RCSfile$
