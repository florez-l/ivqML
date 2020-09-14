// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "NeuralNetwork.h"

// -------------------------------------------------------------------------
template< class _TScalar >
NeuralNetwork< _TScalar >::
NeuralNetwork( )
{
}

// -------------------------------------------------------------------------
template< class _TScalar >
NeuralNetwork< _TScalar >::
NeuralNetwork( const Self& other )
{
  this->m_Layers.clear( );
  this->m_Layers.insert(
    this->m_Layers.begin( ), other.m_Layers.begin( ), other.m_Layers.end( )
    );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename NeuralNetwork< _TScalar >::
Self& NeuralNetwork< _TScalar >::
operator=( const Self& other )
{
  this->m_Layers.clear( );
  this->m_Layers.insert(
    this->m_Layers.begin( ), other.m_Layers.begin( ), other.m_Layers.end( )
    );
  return( *this );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
add( unsigned int i, unsigned int o, const TActivation& f )
{
  this->add( TLayer( i, o, f ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
add( const TMatrix& w, const TColVector& b, const TActivation& f )
{
  this->add( TLayer( w, b, f ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
add( const TLayer& l )
{
  if( this->m_Layers.size( ) > 0 )
    assert( l.input_size( ) == this->m_Layers.back( ).output_size( ) );
  this->m_Layers.push_back( l );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
init( bool randomly )
{
  for( TLayer& l: this->m_Layers )
    l.init( randomly );
}

// -------------------------------------------------------------------------
template< class _TScalar >
typename NeuralNetwork< _TScalar >::
TColVector NeuralNetwork< _TScalar >::
operator()( const TRowVector& x ) const
{
  assert( this->m_Layers.size( ) > 2 );

  auto lIt = this->m_Layers.begin( );
  TColVector z = ( *lIt )( x.transpose( ) );
  for( lIt++; lIt != this->m_Layers.end( ); ++lIt )
    z = ( *lIt )( z );
  return( z );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
_ReadFrom( std::istream& i )
{
  /* TODO
     unsigned int N;
     i >> N;
     for( unsigned int n = 0; n < N; ++n )
     {
     TLayer l;
     i >> l;
     this->add( l );
     } // end for
  */
}

// -------------------------------------------------------------------------
template< class _TScalar >
void NeuralNetwork< _TScalar >::
_CopyTo( std::ostream& o ) const
{
  /* TODO
     o << this->m_Layers.size( ) << std::endl;
     for( const TLayer& l: this->m_Layers )
     o << l << std::endl;
  */
}

// -------------------------------------------------------------------------
template class NeuralNetwork< float >;
template class NeuralNetwork< double >;
template class NeuralNetwork< long double >;

/* TODO
   protected:
   std::vector< TNeuralNetwork > m_NeuralNetworks;

   public:
   ///!
   friend std::istream operator>>( std::istream& i, Self& l )
   {
   l._ReadFrom( i );
   return( i );
   }

   ///!
   friend std::ostream operator<<( std::ostream& o, const Self& l )
   {
   l._CopyTo( o );
   return( o );
   }
   };
   #endif // __PUJ_ML__NeuralNetwork__h__
*/

// eof - $RCSfile$
