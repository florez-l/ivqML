// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

/* TODO
   #include <algorithm>
   #include <cctype>
   #include <stdexcept>
*/

#include <ivqML/Model/NeuralNetwork/FeedForward.h>

// -------------------------------------------------------------------------
template< class _TScl >
ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
FeedForward( )
  : Superclass( TNat( 0 ) )
{
}

// -------------------------------------------------------------------------
template< class _TScl >
bool ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
has_backpropagation( ) const
{
  return( true );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
TNat ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
number_of_inputs( ) const
{
  if( this->m_L.size( ) > 0 )
    return( this->m_L[ 0 ] );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _TScl >
void ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
set_number_of_inputs( const TNat& p )
{
  this->m_L.clear( );
  this->m_L.push_back( p );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
TNat ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
number_of_outputs( ) const
{
  if( this->m_L.size( ) > 0 )
    return( this->m_L.back( ) );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
TNat ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
buffer_size( ) const
{
  TNat s = this->m_L[ 0 ];
  for( TNat i = 1; i < this->m_L.size( ); ++i )
    s += ( this->m_L[ i ] << 1 );
  return( s );
}

// -------------------------------------------------------------------------
template< class _TScl >
void ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
set_number_of_parameters( const TNat& p )
{
  // WARNING: do nothing!
}

// -------------------------------------------------------------------------
template< class _TScl >
void ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
add_layer( const TNat& i )
{
  this->set_number_of_inputs( i );
}

// -------------------------------------------------------------------------
template< class _TScl >
void ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
add_layer( const TNat& o, const std::string& a )
{
  this->m_L.push_back( o );
  // TODO: this->m_A.push_back( std::make_pair( a, TActivationFactory::New( a ) ) );
}

// -------------------------------------------------------------------------
template< class _TScl >
typename ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
TNat ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
number_of_layers( ) const
{
  if( this->m_L.size( ) > 0 )
    return( this->m_L.size( ) - 1 );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _TScl >
void ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
init( )
{
  /* TODO
     this->m_L.shrink_to_fit( );
     this->m_O.clear( );

     TNat P = 0;
     for( TNat l = 1; l < this->m_L.size( ); ++l )
     {
     TNat i = this->m_L[ l - 1 ];
     TNat o = this->m_L[ l ];

     this->m_O.push_back( P );
     this->m_O.push_back( P + ( o * i ) );

     P += o * ( i + 1 );
     } // end for
     this->m_O.shrink_to_fit( );

     this->Superclass::set_number_of_parameters( P );
     this->random_fill( );
  */
}

// -------------------------------------------------------------------------
/* TODO
   template< class _TScl >
   typename ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
   TMatMap ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
   W( const TNat& l )
   {
   if( 1 <= l && l < this->m_L.size( ) )
   return(
   this->matrix(
   this->m_L[ l ], this->m_L[ l - 1 ],
   this->m_O[ ( l - 1 ) >> 1 ]
   )
   );
   else
   return( TMatMap( nullptr, 0, 0 ) );
   }

   // -------------------------------------------------------------------------
   template< class _TScl >
   typename ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
   TMatCMap ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
   W( const TNat& l ) const
   {
   if( 1 <= l && l < this->m_L.size( ) )
   return(
   this->matrix(
   this->m_L[ l ], this->m_L[ l - 1 ],
   this->m_O[ ( l - 1 ) >> 1 ]
   )
   );
   else
   return( TMatCMap( nullptr, 0, 0 ) );
   }

   // -------------------------------------------------------------------------
   template< class _TScl >
   typename ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
   TColMap ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
   B( const TNat& l )
   {
   if( 1 <= l && l < this->m_L.size( ) )
   return(
   this->column(
   this->m_L[ l ], this->m_O[ ( ( l - 1 ) >> 1 ) + 1 ]
   )
   );
   else
   return( TColMap( nullptr, 0, 0 ) );
   }

   // -------------------------------------------------------------------------
   template< class _TScl >
   typename ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
   TColCMap ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
   B( const TNat& l ) const
   {
   if( 1 <= l && l < this->m_L.size( ) )
   return(
   this->column(
   this->m_L[ l ], this->m_O[ ( ( l - 1 ) >> 1 ) + 1 ]
   )
   );
   else
   return( TColCMap( nullptr, 0, 0 ) );
   }
*/

// -------------------------------------------------------------------------
template< class _TScl >
void ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
_from_stream( std::istream& i )
{
  /* TODO
     TNat L, in, out;
     std::string a;

     i >> L >> in >> out >> a;
     this->add_layer( in );
     this->add_layer( out, a );
     for( TNat l = 1; l < L; ++l )
     {
     i >> out >> a;
     this->add_layer( out, a );
     } // end for
     this->init( );

     i >> a;
     std::transform(
     a.begin( ), a.end( ), a.begin( ),
     []( unsigned char c ){ return( std::tolower( c ) ); }
     );
  
     if( a == "random" )
     {
     // Do nothing since init() randomly fills
     }
     else if( a == "zeros" )
     {
     std::transform(
     this->m_P.data( ), this->m_P.data( ) + this->m_P.size( ),
     this->m_P.data( ),
     []( const TScl& v ){ return( TScl( 0 ) ); }
     );
     }
     else if( a == "ones" )
     {
     std::transform(
     this->m_P.data( ), this->m_P.data( ) + this->m_P.size( ),
     this->m_P.data( ),
     []( const TScl& v ){ return( TScl( 1 ) ); }
     );
     }
     else
     {
     TNat P = std::atoi( a.c_str( ) );
     if( P != this->m_P.size( ) )
     throw std::length_error( "Length mismatch while reading model." );
     for( TNat p = 0; p < P; ++p )
     i >> this->m_P( p );
     } // end if
  */
}

// -------------------------------------------------------------------------
template< class _TScl >
void ivqML::Model::NeuralNetwork::FeedForward< _TScl >::
_to_stream( std::ostream& o ) const
{
  /* TODO
     TNat L = this->number_of_layers( );
     o << L << " " << this->m_L[ 0 ] << std::endl;
     for( TNat l = 0; l < L; ++l )
     o
     << this->m_L[ l + 1 ] << " " << this->m_A[ l ].first
     << std::endl;
     this->Superclass::_to_stream( o );
  */
}

// -------------------------------------------------------------------------
namespace ivqML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      template class ivqML_EXPORT FeedForward< float >;
      /* TODO
         template class ivqML_EXPORT FeedForward< double >;
         template class ivqML_EXPORT FeedForward< long double >;
      */
    } // end namespace
  } // end namespace
} // end namespace

// eof - $RCSfile$
