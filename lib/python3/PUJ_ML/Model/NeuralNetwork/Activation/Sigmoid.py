## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Base import *

'''
'''
class Sigmoid( Base ):

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  '''
  def __call__( self, z, d = False ):
    s = 1.0 / ( 1.0 + numpy.exp( -z ) )
    if d:
      return s * ( 1.0 - s )
    else:
      return s
    # end if
  # end def

# end class

## eof - $RCSfile$
