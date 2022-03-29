## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

from abc import ABC, abstractmethod

'''
'''
class Base( ABC ):

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  '''
  def __str__( self ):
    return type( self ).__name__
  # end def

  '''
  '''
  @abstractmethod
  def __call__( self, z, d = False ):
    return None
  # end def

# end class

## eof - $RCSfile$
