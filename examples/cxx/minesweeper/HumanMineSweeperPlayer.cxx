// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "HumanMineSweeperPlayer.h"
#include <cctype>
#include <iostream>
#include <string>

// -------------------------------------------------------------------------
HumanMineSweeperPlayer::
HumanMineSweeperPlayer( )
  : MineSweeperPlayerBase( )
{
}

// -------------------------------------------------------------------------
void HumanMineSweeperPlayer::
play( unsigned int& i, unsigned int& j )
{
  std::string input;
  std::cout << "Choose a cell: " << std::ends;
  std::cin >> input;
  i = std::atoi( input.substr( 0, input.size( ) - 1 ).c_str( ) ) - 1;
  j = std::tolower( input[ input.size( ) - 1 ] ) - 'a';
}

// eof - $RCSfile$
