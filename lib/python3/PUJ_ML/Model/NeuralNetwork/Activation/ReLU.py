## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Base import *

'''
'''
class ReLU( Base ):

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  '''
  def __call__( self, z, d = False ):
    if d:
      return ( z >= 0.0 ).astype( z.dtype )
    else:
      return numpy.array( z ) * numpy.array( ( z >= 0.0 ).astype( z.dtype ) )
    # end if
  # end def

# end class

## eof - $RCSfile$
