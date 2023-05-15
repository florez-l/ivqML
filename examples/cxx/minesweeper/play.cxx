// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <cctype>
#include <iostream>
#include <string>
#include "MineSweeperBoard.h"
#include "HumanMineSweeperPlayer.h"
#include "RandomMineSweeperPlayer.h"
#include "CPUMineSweeperPlayer.h"
#include "LogisticMineSweeperPlayer.h"

int main( int argc, char** argv )
{
  unsigned int w = 8, h = 8, n = 10;
  std::string t = "human";
  if( argc >= 2 )
    t = argv[ 1 ];
  if( argc >= 3 )
    w = std::atoi( argv[ 2 ] );
  if( argc >= 4 )
    h = std::atoi( argv[ 3 ] );
  if( argc >= 5 )
    n = std::atoi( argv[ 4 ] );

  // Configure board
  MineSweeperBoard* board = new MineSweeperBoard( w, h, n );

  // Configure player
  MineSweeperPlayerBase* player = nullptr;
  if( t == "human" )
    player = new HumanMineSweeperPlayer( );
  else if( t == "random" )
    player = new RandomMineSweeperPlayer( );
  else if( t == "cpu" )
    player = new CPUMineSweeperPlayer( );
  else if( t == "logistic" )
    player = new LogisticMineSweeperPlayer( );
  board->set_player( player );

  while( !( board->have_won( ) ||  board->have_lose( ) ) )
  {
    // Show board
    std::cout << *board << std::endl;

    // Advance one step
    board->step( );
  } // end while

  // Show final result
  std::cout << *board << std::endl;
  if( board->have_won( ) )
    std::cout << "You won!" << std::endl;
  else if( board->have_lose( ) )
    std::cout << "You lose." << std::endl;

  // Free objects
  delete board;
  delete player;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
