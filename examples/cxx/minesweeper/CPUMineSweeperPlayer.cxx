// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "CPUMineSweeperPlayer.h"
#include <algorithm>
#include <random>

// -------------------------------------------------------------------------
CPUMineSweeperPlayer::
CPUMineSweeperPlayer( )
  : MineSweeperPlayerBase( )
{
  this->m_Compare =
    []( const TTuple& a, const TTuple& b ) -> bool
    {
      return( std::get< 0 >( a ) > std::get< 0 >( b ) );
    };
}

// -------------------------------------------------------------------------
void CPUMineSweeperPlayer::
configure( unsigned int w, unsigned int h, unsigned long n )
{
  this->MineSweeperPlayerBase::configure( w, h, n );

  this->m_Patches.resize( w * h, false );

  double p0 = double( n ) / double( w * h );
  for( unsigned int i = 0; i < this->m_Height; ++i )
    for( unsigned int j = 0; j < this->m_Width; ++j )
      this->m_Options.push_back( std::make_tuple( p0, i, j ) );
  this->m_Options.shrink_to_fit( );
  std::random_device rd;
  std::default_random_engine rng( rd( ) );
  std::shuffle( this->m_Options.begin( ), this->m_Options.end( ), rng );
}

// -------------------------------------------------------------------------
void CPUMineSweeperPlayer::
play( unsigned int& i, unsigned int& j )
{
  if( this->m_Options.size( ) > 0 )
  {
    i = std::get< 1 >( this->m_Options.front( ) );
    j = std::get< 2 >( this->m_Options.front( ) );
    while( this->m_Patches[ this->_idx( i, j ) ] )
    {
      std::pop_heap(
        this->m_Options.begin( ), this->m_Options.end( ), this->m_Compare
        );
      this->m_Options.pop_back( );
      i = std::get< 1 >( this->m_Options.front( ) );
      j = std::get< 2 >( this->m_Options.front( ) );
    } // end while
    this->m_Patches[ this->_idx( i, j ) ] = true;
  }
  else
    i = j = 0;
}

// -------------------------------------------------------------------------
void CPUMineSweeperPlayer::
report( const unsigned char& c )
{
  unsigned int i = std::get< 1 >( this->m_Options.front( ) );
  unsigned int j = std::get< 2 >( this->m_Options.front( ) );

  std::pop_heap(
    this->m_Options.begin( ), this->m_Options.end( ), this->m_Compare
    );
  this->m_Options.pop_back( );

  // Get neighborhood
  std::vector< std::pair< unsigned int, unsigned int > > neighborhood;
  for( int k = -1; k <= 1; ++k )
  {
    unsigned int x = i + k;
    if( x < this->m_Height )
    {
      for( int l = -1; l <= 1; ++l )
      {
        if( k != 0 || l != 0 )
        {
          unsigned int y = j + l;
          if( y < this->m_Width )
          {
            if( !( this->m_Patches[ this->_idx( x, y ) ] ) )
            {
              neighborhood.push_back( std::make_pair( x, y ) );
            } // end if
          } // end if
        } // end if
      } // end for
    } // end if
  } // end for

  double p = double( c ) / double( neighborhood.size( ) );
  /* TODO
     if( p == 1 )
     std::cout << "ok" << std::endl;
  */

  for( const auto& n: neighborhood )
  {
    this->m_Options.push_back( std::make_tuple( p, n.first, n.second ) );
    std::push_heap(
      this->m_Options.begin( ), this->m_Options.end( ), this->m_Compare
      );
  } // end for
}

// -------------------------------------------------------------------------
unsigned long long CPUMineSweeperPlayer::
_idx( const unsigned int& i, const unsigned int& j ) const
{
  auto l = (
    ( ( unsigned long long )( i ) * ( unsigned long long )( this->m_Width ) )
    +
    ( unsigned long long )( j )
    );
  return( l );
}

// eof - $RCSfile$

