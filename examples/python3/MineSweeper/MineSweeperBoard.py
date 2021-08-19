## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import random

'''
'''
class MineSweeperBoard:

  '''
  '''
  m_Patches = None
  m_Mines = None
  m_NumberOfMines = 0
  m_Explosion = False

  '''
  '''
  def __init__( self, w, h, n ):
    self.m_Patches = [ [ False for j in range( h ) ] for i in range( w ) ]
    self.m_Mines = [ [ 0 for j in range( h ) ] for i in range( w ) ]
    self.m_NumberOfMines = n
    self.m_Explosion = False

    # Choose positions for mines
    t = [ i for i in range( w * h ) ]
    random.shuffle( t )
    for i in t[ 0: n ]:
      self.m_Mines[ int( i / w ) ][ i % w ] = 9
    # end for

    # Fill remaining cells
    for i in range( len( self.m_Mines ) ):
      for j in range( len( self.m_Mines[ i ] ) ):
        if self.m_Mines[ i ][ j ] == 0:
          for k in range( i - 1, i + 2 ):
            if k >= 0 and k < len( self.m_Mines ):
              for l in range( j - 1, j + 2 ):
                if l >= 0 and l < len( self.m_Mines[ j ] ):
                  if self.m_Mines[ k ][ l ] == 9:
                    self.m_Mines[ i ][ j ] += 1
                  # end if
                # end if
              # end for
            # end if
          # end for
        # end if
      # end for
    # end for

  # end def

  '''
  '''
  def __str__( self ):
    s = "    "
    for k in range( len( self.m_Mines ) ):
      s += "+---"
    # end for
    s += "+\n    "
    for k in range( len( self.m_Mines ) ):
      s += "| " + chr( ord( '1' ) + k ) + " "
    # end for
    s += "|\n"
    for j in range( len( self.m_Mines[ 0 ] ) ):
      for k in range( len( self.m_Mines ) + 1 ):
        s += "+---"
      # end for
      s += "+\n"
      s += "| " + chr( ord( 'A' ) + j ) + " "
      for i in range( len( self.m_Mines ) ):
        s += "| "
        if self.m_Patches[ i ][ j ]:
            if self.m_Mines[ i ][ j ] < 9:
              s += str( self.m_Mines[ i ][ j ] )
            else:
              s += "X"
            # end if
        else:
            s += " "
        # end if
        s += " "
      # end for
      s += "|\n"
    # end for
    for k in range( len( self.m_Mines ) + 1 ):
      s += "+---"
    # end for
    s += "+\n"
    return s
  # end def

  '''
  '''
  def __repr__( self ):
    return self.__str__( )
  # end def

  '''
  '''
  def width( self ):
    return len( self.m_Mines )
  # end def

  '''
  '''
  def number_of_mines( self ):
    return self.m_NumberOfMines
  # end def

  '''
  '''
  def height( self ):
    if len( self.m_Mines ) > 0:
      return len( self.m_Mines[ 0 ] )
    else:
      return 0
    # end if
  # end def

  '''
  '''
  def have_won( self ):
    c = 0
    for i in range( len( self.m_Mines ) ):
      for j in range( len( self.m_Mines[ i ] ) ):
        if not self.m_Patches[ i ][ j ]:
          c += 1
        # end if
      # end for
    # end for
    return c == self.m_NumberOfMines
  # end def

  '''
  '''
  def have_lose( self ):
    return self.m_Explosion
  # end def

  '''
  '''
  def click( self, i, j ):
    if not self.m_Explosion:
      if i < 0 or j < 0 or i >= self.width( ) or j >= self.height( ):
        return 0
      else:
        if not self.m_Patches[ i ][ j ]:
          self.m_Patches[ i ][ j ] = True
        # end if
        if self.m_Mines[ i ][ j ] == 9:
          self.m_Explosion = True
          self.m_Patches = [ [ True for j in range( self.height( ) ) ] for i in range( self.width( ) ) ]
        # end if
        return self.m_Mines[ i ][ j ]
      # end if
    else:
      return self.m_Mines[ i ][ j ]
    # end if
  # end def

# end class

## eof - $RCSfile$
