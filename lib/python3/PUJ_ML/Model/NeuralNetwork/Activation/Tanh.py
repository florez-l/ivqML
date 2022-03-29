## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Base import *

'''
'''
class Tanh( Base ):

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  '''
  def __call__( self, z, d = False ):
    t = numpy.tanh( z )
    if d:
      return 1.0 - ( t ** 2 )
    else:
      return t
    # end if
  # end def

# end class

## eof - $RCSfile$
