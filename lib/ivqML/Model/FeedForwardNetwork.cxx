// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/Model/FeedForwardNetwork.h>

// -------------------------------------------------------------------------
template< class _S >
ivqML::Model::FeedForwardNetwork< _S >::
FeedForwardNetwork( )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
random_fill( )
{
  this->Superclass::random_fill( );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::FeedForwardNetwork< _S >::
TNatural ivqML::Model::FeedForwardNetwork< _S >::
number_of_inputs( ) const
{
  if( this->m_S.size( ) > 0 )
    return( this->m_S[ 0 ] );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::FeedForwardNetwork< _S >::
TNatural ivqML::Model::FeedForwardNetwork< _S >::
number_of_outputs( ) const
{
  if( this->m_S.size( ) > 0 )
    return( this->m_S.back( ) );
  else
    return( 0 );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
set_number_of_parameters( const TNatural& p )
{
  // WARNING: do nothing!
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
add_layer( const TNatural& i, const TNatural& o, const std::string& a )
{
  this->m_S.clear( );
  this->m_F.clear( );

  this->m_S.push_back( i );
  this->m_S.push_back( o );

  this->m_F.push_back( std::make_pair( a, TActivationFactory::New( a ) ) );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
add_layer( const TNatural& o, const std::string& a )
{
  this->m_S.push_back( o );
  this->m_F.push_back( std::make_pair( a, TActivationFactory::New( a ) ) );
}

// -------------------------------------------------------------------------
template< class _S >
typename ivqML::Model::FeedForwardNetwork< _S >::
TNatural ivqML::Model::FeedForwardNetwork< _S >::
number_of_layers( ) const
{
  return( this->m_S.size( ) - 1 );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
init( )
{
  this->m_S.shrink_to_fit( );

  TNatural P = 0;
  for( TNatural l = 1; l < this->m_S.size( ); ++l )
  {
    TNatural i = this->m_S[ l - 1 ];
    TNatural o = this->m_S[ l ];
    P += o * ( i + 1 );
  } // end for
  this->Superclass::set_number_of_parameters( P );

  this->m_W.clear( );
  this->m_B.clear( );
  TNatural s = 0;
  TScalar* b = this->begin( );
  for( TNatural l = 1; l < this->m_S.size( ); ++l )
  {
    TNatural i = this->m_S[ l - 1 ];
    TNatural o = this->m_S[ l ];

    this->m_W.push_back( TMap( b + s, i, o ) );
    this->m_B.push_back( TMap( b + s + ( i * o ), 1, o ) );

    s += o * ( i + 1 );
  } // end for
  this->m_W.shrink_to_fit( );
  this->m_B.shrink_to_fit( );

  this->random_fill( );
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
_from_stream( std::istream& i )
{
  /* TODO
     TNatural L = this->number_of_layers( );
     o << L << " " << this->m_S[ 0 ] << std::endl;
     for( TNatural l = 0; l < L; ++l )
     o
     << this->m_S[ l + 1 ] << " "
     << this->m_F[ l ].first << std::endl;
     this->Superclass::_to_stream( o );
  */
}

// -------------------------------------------------------------------------
template< class _S >
void ivqML::Model::FeedForwardNetwork< _S >::
_to_stream( std::ostream& o ) const
{
  TNatural L = this->number_of_layers( );
  o << L << " " << this->m_S[ 0 ] << std::endl;
  for( TNatural l = 0; l < L; ++l )
    o
      << this->m_S[ l + 1 ] << " "
      << this->m_F[ l ].first << std::endl;
  this->Superclass::_to_stream( o );
}

// -------------------------------------------------------------------------
template class ivqML_EXPORT ivqML::Model::FeedForwardNetwork< float >;
template class ivqML_EXPORT ivqML::Model::FeedForwardNetwork< double >;
template class ivqML_EXPORT ivqML::Model::FeedForwardNetwork< long double >;

// eof - $RCSfile$
