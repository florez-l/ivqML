// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__FeedForward__hxx__
#define __ivqML__Model__NeuralNetwork__FeedForward__hxx__

#include <cstdlib>

// -------------------------------------------------------------------------
template< class _TScalar >
template< class _TInputX >
auto ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
eval( const Eigen::EigenBase< _TInputX >& iX ) const
{
  TRow buffer = TRow( this->buffer_size( ) * iX.cols( ) );

  std::cout << "-------------> " << buffer.size( ) << std::endl;
  this->_eval( iX, buffer.data( ) );

  return( buffer.data( ) + ( buffer.size( ) - ( this->m_S.back( ) * iX.cols( ) ) ) );

  // Computation buffer
  /* TODO
     TNatural m = iX.cols( );
     TNatural s = this->buffer_size( ) * m;
     TScalar* buffer = iB;
     if( iB == nullptr )
     buffer =
     reinterpret_cast< TScalar* >( std::calloc( s, sizeof( TScalar ) ) );

     // Input layer
     TMatMap( buffer, this->m_Layers[ 0 ], m )
     =
     iX.derived( ).template cast< TScalar >( );

     // Loop
     TNatural as = 0, zs = this->m_Layers[ 0 ] * m;
     TNatural ws = 0, bs = this->m_Layers[ 0 ] * this->m_Layers[ 1 ];
     TNatural L = this->number_of_layers( );
     for( TNatural l = 0; l < L; ++l )
     {
     TMatMap Z( buffer + zs, this->m_Layers[ l + 1 ], m );
     TMatMap A( buffer + as, this->m_Layers[ l ], m );
     TMatMap R(
     buffer + ( zs + ( this->m_Layers[ l + 1 ] * m ) ),
     this->m_Layers[ l + 1 ], m
     );
     auto W = this->_matrix( this->m_Layers[ l + 1 ], this->m_Layers[ l ], ws );
     auto B = this->_column( this->m_Layers[ l + 1 ], bs );

     Z = ( W * A ).colwise( ) + B;
     this->m_F[ l ].second( R, Z, false );

     if( l < L - 1 )
     {
     as += ( this->m_Layers[ l ] + this->m_Layers[ l + 1 ] ) * m;
     zs += ( this->m_Layers[ l + 1 ] + this->m_Layers[ l + 1 ] ) * m;
     ws = bs + this->m_Layers[ l + 1 ];
     bs = ws + ( this->m_Layers[ l + 2 ] * this->m_Layers[ l + 1 ] );
     } // end if
     } // end for

     TMat res = TMatMap( buffer + as, this->m_Layers[ L ], m );
     if( iB == nullptr )
     std::free( buffer );
     return( res );
  */
}

// -------------------------------------------------------------------------
template< class _TScalar >
template< class _TInputX >
void ivqML::Model::NeuralNetwork::FeedForward< _TScalar >::
_eval( const Eigen::EigenBase< _TInputX >& iX, TScalar* buffer ) const
{
}

// -------------------------------------------------------------------------
/* TODO
   template< class _S >
   template< class _X, class _Y >
   void ivqML::Model::NeuralNetwork::FeedForward< _S >::
   cost(
   TScalar* bG,
   const Eigen::EigenBase< _X >& iX,
   const Eigen::EigenBase< _Y >& iY,
   TScalar* J, TScalar* iB
   ) const
   {
   // Computation buffer
   TNatural m = iX.cols( );
   TNatural nparams = this->number_of_parameters( );
   TNatural bsize = this->m_BSize * m;
   TScalar* buffer = iB;
   if( iB == nullptr )
   buffer =
   reinterpret_cast< TScalar* >( std::malloc( sizeof( TScalar ) * bsize ) );

   // Forward propagation
   this->evaluate( iX, buffer );

   // Memory ranges
   TNatural L = this->number_of_layers( );
   TNatural as = bsize - ( this->m_S[ L ] * m );
   TNatural zs = as - ( this->m_S[ L ] * m );
   TNatural bs = nparams - this->m_S[ L ];
   TNatural ws = bs - ( this->m_S[ L ] * this->m_S[ L - 1 ] );

   // Last layer delta
   TMap( buffer + as, this->m_S[ L ], m )
   -=
   iY.derived( ).template cast< TScalar >( );

   // Remaining layers
   for( TNatural l = L; l > 0; --l )
   {
   TMap D( buffer + as, this->m_S[ l ], m );
   as -= ( this->m_S[ l - 1 ] + this->m_S[ l ] ) * m;
   zs = as - ( this->m_S[ l - 1 ] * m );
   TMap E( buffer + as, this->m_S[ l - 1 ], m );

   // Update derivatives
   TMap( bG + bs, this->m_S[ l ], 1 )
   =
   D.rowwise( ).mean( );
   TMap( bG + ws, this->m_S[ l ], this->m_S[ l - 1 ] )
   =
   ( D * E.transpose( ) ) / TScalar( m );

   // Update delta if there is more back layers
   if( l > 1 )
   {
   bs = ws - this->m_S[ l - 1 ];
   ws = bs - ( this->m_S[ l - 1 ] * this->m_S[ l - 2 ] );
   E = this->m_W[ l - 1 ].transpose( ) * D;

   TMap Z( buffer + zs, this->m_S[ l - 1 ], m );
   this->m_F[ l - 1 ].second( Z, Z, true );
   E.array( ) *= Z.array( );
   } // end if
   } // end for

   // Free buffer
   if( iB != nullptr )
   std::free( buffer );
   }
*/

#endif // __ivqML__Model__NeuralNetwork__FeedForward__hxx__

// eof - $RCSfile$
