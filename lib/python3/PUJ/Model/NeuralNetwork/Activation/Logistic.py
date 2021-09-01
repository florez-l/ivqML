## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Base import *

'''
'''
class Logistic( Base ):

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  '''
  def __call__( self, z, derivative = False ):
    s = 1.0 / ( 1.0 + numpy.exp( -z ) )
    if derivative:
      return s * ( 1.0 - s )
    else:
      return s
    # end if
  # end def

# end class

## eof - $RCSfile$
