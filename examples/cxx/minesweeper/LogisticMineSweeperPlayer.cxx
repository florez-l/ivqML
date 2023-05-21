// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "LogisticMineSweeperPlayer.h"
#include <algorithm>
#include <random>

// -------------------------------------------------------------------------
LogisticMineSweeperPlayer::
LogisticMineSweeperPlayer( )
  : MineSweeperPlayerBase( )
{
}

// -------------------------------------------------------------------------
void LogisticMineSweeperPlayer::
configure( unsigned int w, unsigned int h, unsigned long n )
{
  this->MineSweeperPlayerBase::configure( w, h, n );

  this->m_Model.init( 8 );
  this->m_Inputs.resize( this->m_Width * this->m_Height, 8 );

  unsigned long long r = 0;
  for( unsigned int i = 0; i < this->m_Height; ++i )
  {
    for( unsigned int j = 0; j < this->m_Width; ++j )
    {
      unsigned long long c = 0;
      for( int k = -1; k <= 1; ++k )
      {
        unsigned int x = i + k;
        for( int l = -1; l <= 1; ++l )
        {
          if( k != 0 || l != 0 )
          {
            unsigned int y = j + l;
            if( x < this->m_Height && y < this->m_Width )
              this->m_Inputs( r, c ) = 9;
            else
              this->m_Inputs( r, c ) = 0;
            c++;
          } // end if
        } // end for
      } // end for
      r++;
    } // end for
  } // end for

  // Shuffle all possible inputs
  std::vector< unsigned long long > idx( this->m_Width * this->m_Height );
  std::iota( idx.begin( ), idx.end( ), 0 );
  std::random_device rd;
  std::default_random_engine rng( rd( ) );
  std::shuffle( idx.begin( ), idx.end( ), rng );

  // Assign them
  this->m_Options.clear( );
  for( const unsigned long long& i: idx )
    this->m_Options.push_back(
      std::make_pair(
        ( unsigned int )( i / this->m_Width ),
        ( unsigned int )( i % this->m_Width )
        )
      );
  this->m_Options.shrink_to_fit( );
  this->m_Inputs = this->m_Inputs( idx, Eigen::all );
}

// -------------------------------------------------------------------------
void LogisticMineSweeperPlayer::
play( unsigned int& i, unsigned int& j )
{
  if( this->m_Options.size( ) > 0 )
  {
    TModel::TCol Z;
    this->m_Model.evaluate( Z, this->m_Inputs );
    Z.minCoeff( &( this->m_Choice ) );
    i = this->m_Options[ this->m_Choice ].first;
    j = this->m_Options[ this->m_Choice ].second;
  }
  else
    i = j = 0;
}

// -------------------------------------------------------------------------
void LogisticMineSweeperPlayer::
report( const unsigned char& c )
{
  // Last choice
  unsigned int i = this->m_Options[ this->m_Choice ].first;
  unsigned int j = this->m_Options[ this->m_Choice ].second;

  // Remove option
  unsigned int rows = this->m_Inputs.rows( ) - 1;
  unsigned int cols = this->m_Inputs.cols( );
  unsigned int keep = rows - this->m_Choice;
  if( this->m_Choice < rows )
    this->m_Inputs.block( this->m_Choice, 0, keep, cols )
      =
      this->m_Inputs.block( this->m_Choice + 1, 0, keep, cols );
  this->m_Inputs.conservativeResize( rows, cols );
  this->m_Options.erase( this->m_Options.begin( ) + this->m_Choice );
}

// -------------------------------------------------------------------------
unsigned long long LogisticMineSweeperPlayer::
_idx( const unsigned int& i, const unsigned int& j ) const
{
  auto l = (
    (
      ( unsigned long long )( i )
      *
      ( unsigned long long )( this->m_Width )
      )
    +
    ( unsigned long long )( j )
    );
  return( l );
}

// eof - $RCSfile$
