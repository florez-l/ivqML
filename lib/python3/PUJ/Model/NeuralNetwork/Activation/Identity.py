## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Base import *

'''
'''
class Identity( Base ):

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  '''
  def __call__( self, z, derivative = False ):
    if derivative:
      return numpy.ones( z.shape )
    else:
      return z
    # end if
  # end def

# end class

## eof - $RCSfile$
