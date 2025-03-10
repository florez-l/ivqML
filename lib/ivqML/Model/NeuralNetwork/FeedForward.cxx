// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <random>
#include <ivqML/Model/NeuralNetwork/FeedForward.h>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
FeedForward( )
  : Superclass( 0 )
{
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
~FeedForward( )
{
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
set_input_layer( const TNatural& i, const TNatural& o, TActivation a )
{
  this->m_N.clear( );
  this->m_W.clear( );
  this->m_B.clear( );
  this->m_F.clear( );

  this->m_N.push_back( i );
  this->m_N.push_back( o );
  this->m_F.push_back( a );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
set_input_layer( const TNatural& i, const TNatural& o, const std::string& a )
{
  this->set_input_layer( i, o, TFunctions::Get( a ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
add_layer( const TNatural& o, TActivation a )
{
  this->m_N.push_back( o );
  this->m_F.push_back( a );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
add_layer( const TNatural& o, const std::string& a )
{
  this->add_layer( o, TFunctions::Get( a ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
typename ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TNatural ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
number_of_layers( ) const
{
  return( this->m_W.size( ) );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
const typename ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TNatural& ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
input_size( const TNatural& l ) const
{
  static const TNatural _0 = TNatural( 0 );
  if( l < this->m_N.size( ) )
    return( this->m_N[ l ] );
  else
    return( _0 );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
const typename ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TNatural& ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
output_size( const TNatural& l ) const
{
  static const TNatural _0 = TNatural( 0 );
  if( this->m_N.size( ) > 0 )
  {
    if( l == 0 )
      return( this->m_N[ this->m_N.size( ) - 1 ] );
    else
    {
      TNatural i = l + 1;
      if( i < this->m_N.size( ) )
        return( this->m_N[ i ] );
      else
        return( _0 );
    }
  }
  else
    return( _0 );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
typename ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TReal& ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
operator[]( std::initializer_list< TNatural > i )
{
  static TReal _0 = TReal( 0 );

  TNatural L = this->number_of_layers( );
  if( i.size( ) == 2 )
  {
    auto j = i.begin( );
    TNatural l = *( j++ );
    TNatural r = *( j++ );
    if( l < L )
    {
      if( r < this->m_B[ l ].rows( ) )
        return( this->m_B[ l ]( r, 0 ) );
      else
      {
        _0 = TReal( 0 );
        return( _0 );
      } // end if
    }
    else
    {
      _0 = TReal( 0 );
      return( _0 );
    } // end if
  }
  else if( i.size( ) == 3 )
  {
    auto j = i.begin( );
    TNatural l = *( j++ );
    TNatural r = *( j++ );
    TNatural c = *( j++ );
    if( l < L )
    {
      if( r < this->m_W[ l ].rows( ) )
      {
        if( c < this->m_W[ l ].cols( ) )
          return( this->m_W[ l ]( r, c ) );
        else
        {
          _0 = TReal( 0 );
          return( _0 );
        } // end if
      }
      else
      {
        _0 = TReal( 0 );
        return( _0 );
      } // end if
    }
    else
    {
      _0 = TReal( 0 );
      return( _0 );
    } // end if
  }
  else
  {
    _0 = TReal( 0 );
    return( _0 );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
const typename ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
TReal& ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
operator[]( std::initializer_list< TNatural > i ) const
{
  static const TReal _0 = TReal( 0 );

  TNatural L = this->number_of_layers( );
  if( i.size( ) == 2 )
  {
    auto j = i.begin( );
    TNatural l = *( j++ );
    TNatural r = *( j++ );
    if( l < L )
    {
      if( r < this->m_B[ l ].rows( ) )
        return( this->m_B[ l ]( r, 0 ) );
      else
        return( _0 );
    }
    else
      return( _0 );
  }
  else if( i.size( ) == 3 )
  {
    auto j = i.begin( );
    TNatural l = *( j++ );
    TNatural r = *( j++ );
    TNatural c = *( j++ );
    if( l < L )
    {
      if( r < this->m_W[ l ].rows( ) )
      {
        if( c < this->m_W[ l ].cols( ) )
          return( this->m_W[ l ]( r, c ) );
        else
          return( _0 );
      }
      else
        return( _0 );
    }
    else
      return( _0 );
  }
  else
    return( _0 );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
init( )
{
  // Reserve space for all parameters
  TNatural L = this->m_N.size( ) - 1;
  TNatural N = 0;
  for( TNatural l = 0; l < L; ++l )
    N += ( this->m_N[ l ] + 1 ) * this->m_N[ l + 1 ];
  this->_resize( N );

  // Map parameters memory
  TReal* b = this->m_P;
  for( TNatural l = 0; l < L; ++l )
  {
    TNatural i = this->m_N[ l ];
    TNatural o = this->m_N[ l + 1 ];

    this->m_W.push_back( TMatMap( b, i, o ) );
    b += i * o;
    this->m_B.push_back( TRowMap( b, 1, o ) );
    b += o;
  } // end for

  // Init some random parameters
  std::random_device rd;
  std::mt19937 rg( rd( ) );
  std::uniform_real_distribution< TReal > rdis(
    std::numeric_limits< TReal >::epsilon( ),
    TReal( 1 )
    );
  std::generate(
    this->m_P, this->m_P + this->m_S,
    [&]( ) -> TReal
    {
      return( ( TReal( 2 ) * rdis( rg ) ) - TReal( 1 ) );
    }
    );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
_eval( TReal* Ab, TReal* Zb, const TNatural& M, bool keep_AZ ) const
{
  TNatural L = this->number_of_layers( );
  for( TNatural l = 0; l < L; ++l )
  {
    TNatural i = this->m_N[ l ];
    TNatural o = this->m_N[ l + 1 ];

    TMatMap Z( Zb, M, o );
    Z = ( TMatMap( Ab, M, i ) * this->m_W[ l ] ).rowwise( ) + this->m_B[ l ];
    TMatMap A( Ab, M, o );
    this->m_F[ l ]( A, Z, false );
  } // end for
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void  ivqML::Model::NeuralNetwork::FeedForward< _TReal, _TNatural >::
_to_stream( std::ostream& o ) const
{
  this->Superclass::_to_stream( o );
}

// -------------------------------------------------------------------------
namespace ivqML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      template class ivqML_EXPORT FeedForward< float, unsigned int >;
      template class ivqML_EXPORT FeedForward< float, unsigned long >;
      template class ivqML_EXPORT FeedForward< float, unsigned long long >;

      template class ivqML_EXPORT FeedForward< double, unsigned int >;
      template class ivqML_EXPORT FeedForward< double, unsigned long >;
      template class ivqML_EXPORT FeedForward< double, unsigned long long >;

      template class ivqML_EXPORT FeedForward< long double, unsigned int >;
      template class ivqML_EXPORT FeedForward< long double, unsigned long >;
      template class ivqML_EXPORT FeedForward< long double, unsigned long long >;
    } // end namespace
  } // end namespace
} // end namespace

// eof - $RCSfile$
