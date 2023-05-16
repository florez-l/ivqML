// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "MineSweeperBoard.h"
#include "MineSweeperPlayerBase.h"
#include <algorithm>
#include <random>

// -------------------------------------------------------------------------
MineSweeperBoard::
MineSweeperBoard( unsigned int w, unsigned int h, unsigned long n )
{
  this->m_Width = w;
  this->m_Height = h;
  this->m_Patches.resize( w * h, false );
  this->m_Mines.resize( w * h, 0 );
  this->m_NumberOfMines = n;
  this->m_Explosion = false;

  // Choose positions for mines
  for( unsigned long long i = 0; i < this->m_NumberOfMines; ++i )
    this->m_Mines[ i ] = 9;
  std::random_device rd;
  std::default_random_engine rng( rd( ) );
  std::shuffle( this->m_Mines.begin( ), this->m_Mines.end( ), rng );

  // Fill remaining cells with mine counts
  for( unsigned int i = 0; i < this->m_Height; ++i )
  {
    for( unsigned int j = 0; j < this->m_Width; ++j )
    {
      if( this->m_Mines[ this->_idx( i, j ) ] == 9 )
      {
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
                  if( this->m_Mines[ this->_idx( x, y ) ] < 9 )
                    this->m_Mines[ this->_idx( x, y ) ] += 1;
              } // end if
            } // end for
          } // end if
        } // end for
      } // end if
    } // end for
  } // end for
}

// -------------------------------------------------------------------------
void MineSweeperBoard::
set_player( MineSweeperPlayerBase* p )
{
  this->m_Player = p;
  this->m_Player->configure(
    this->m_Width, this->m_Height, this->m_NumberOfMines
    );
}

// -------------------------------------------------------------------------
const unsigned int& MineSweeperBoard::
width( ) const
{
  return( this->m_Width );
}

// -------------------------------------------------------------------------
const unsigned int& MineSweeperBoard::
height( ) const
{
  return( this->m_Height );
}

// -------------------------------------------------------------------------
const unsigned long long& MineSweeperBoard::
number_of_mines( ) const
{
  return( this->m_NumberOfMines );
}

// -------------------------------------------------------------------------
bool MineSweeperBoard::
have_won( ) const
{
  unsigned long long c = 0;
  for( unsigned int i = 0; i < this->m_Height; ++i )
    for( unsigned int j = 0; j < this->m_Width; ++j )
      if( !( this->m_Patches[ this->_idx( i, j ) ] ) )
        c++;
  return( c == this->m_NumberOfMines );
}

// -------------------------------------------------------------------------
bool MineSweeperBoard::
have_lose( ) const
{
  return( this->m_Explosion );
}

// -------------------------------------------------------------------------
unsigned char MineSweeperBoard::
click( unsigned int i, unsigned int j )
{
  if( this->m_Explosion )
    return( this->m_Mines[ this->_idx( i, j ) ] );
  if( i < 0 || j < 0 || i >= this->height( ) || j >= this->width( ) )
    return( 0 );

  if( !( this->m_Patches[ this->_idx( i, j ) ] ) )
  {
    this->m_Patches[ this->_idx( i, j ) ] = true;
    if( this->m_Mines[ this->_idx( i, j ) ] == 9 )
    {
      this->m_Explosion = true;
      for( unsigned int p = 0; p < this->m_Patches.size( ); ++p )
        this->m_Patches[ p ] = true;
    } // end if
  } // end if
  return( this->m_Mines[ this->_idx( i, j ) ] );
}

// -------------------------------------------------------------------------
void MineSweeperBoard::
step( )
{
  // Choose a cell
  unsigned int i, j;
  this->m_Player->play( i, j );

  // Click
  this->m_Player->report( this->click( i, j ) );
}

// -------------------------------------------------------------------------
void MineSweeperBoard::
_to_stream( std::ostream& out ) const
{
  out << "+";
  for( unsigned int j = 0; j <= this->m_Width; ++j )
    out << "===+";
  out << std::endl;
  out << "|   |";
  for( unsigned int j = 0; j < this->m_Width; ++j )
    out << " " << char( int( 'A' ) + j ) << " |";
  out << std::endl;
  out << "+";
  for( unsigned int j = 0; j <= this->m_Width; ++j )
    out << "===+";
  out << std::endl;
  for( unsigned int i = 0; i < this->m_Height; ++i )
  {
    out << "|*" << i + 1 << "*|";
    for( unsigned int j = 0; j < this->m_Width; ++j )
    {
      out << " ";
      if( this->m_Patches[ this->_idx( i, j ) ] )
      {
      if( this->m_Mines[ this->_idx( i, j ) ] < 9 )
        out << int( this->m_Mines[ this->_idx( i, j ) ] );
      else
        out << "X";
      out << " |";
      }
      else
        out << "  |";
    } // end for
    out << std::endl;
    out << "+";
    for( unsigned int j = 0; j <= this->m_Width; ++j )
      out << "---+";
    out << std::endl;
  } // end for
}

// -------------------------------------------------------------------------
unsigned long long MineSweeperBoard::
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
