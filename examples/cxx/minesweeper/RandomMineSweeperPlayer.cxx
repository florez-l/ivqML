// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "RandomMineSweeperPlayer.h"
#include <algorithm>
#include <random>

// -------------------------------------------------------------------------
RandomMineSweeperPlayer::
RandomMineSweeperPlayer( )
  : MineSweeperPlayerBase( )
{
}

// -------------------------------------------------------------------------
void RandomMineSweeperPlayer::
configure( unsigned int w, unsigned int h, unsigned long n )
{
  this->MineSweeperPlayerBase::configure( w, h, n );

  for( unsigned int i = 0; i < this->m_Height; ++i )
    for( unsigned int j = 0; j < this->m_Width; ++j )
      this->m_Options.push_back( std::make_pair( i, j ) );
  this->m_Options.shrink_to_fit( );
}

// -------------------------------------------------------------------------
void RandomMineSweeperPlayer::
play( unsigned int& i, unsigned int& j )
{
  if( this->m_Options.size( ) > 0 )
  {
    std::random_device rd;
    std::default_random_engine rng( rd( ) );
    std::shuffle( this->m_Options.begin( ), this->m_Options.end( ), rng );
    i = this->m_Options.back( ).first;
    j = this->m_Options.back( ).second;
    this->m_Options.pop_back( );
  }
  else
    i = j = 0;
}

// eof - $RCSfile$
