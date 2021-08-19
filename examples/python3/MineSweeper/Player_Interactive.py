## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import sys
from MineSweeperBoard import *

## -------------------------------------------------------------------------
if len( sys.argv ) < 4:
  print( "Usage: python", sys.argv[ 0 ], "width height mines" )
  sys.exit( 1 )
# end if
w = int( sys.argv[ 1 ] )
h = int( sys.argv[ 2 ] )
m = int( sys.argv[ 3 ] )
board = MineSweeperBoard( w, h, m )

while not board.have_won( ) and not board.have_lose( ):
  print( board )
  c = input( "Choose a cell: " ).lower( )
  if len( c ) == 2:
    i = ord( c[ 0 ] ) - ord( 'a' )
    j = ord( c[ 1 ] ) - ord( '1' )
    board.click( j, i )
  # end if
# end while

print( board )
if board.have_won( ):
  print( "You won!" )
elif board.have_lose( ):
  print( "You lose :-(" )
# end if

## eof - $RCSfile$
