## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Base import *

'''
'''
class BinaryStep( Base ):

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  '''
  def __call__( self, z, derivative = False ):
    if derivative:
      return numpy.zeros( z.shape )
    else:
      return ( z >= 0.0 ).astype( z.dtype )
    # end if
  # end def

# end class

## eof - $RCSfile$
