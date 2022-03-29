## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Base import *

'''
'''
class ArcTan( Base ):

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  '''
  def __call__( self, z, d = False ):
    if d:
      return 1.0 / ( 1.0 + ( z ** 2 ) )
    else:
      return numpy.arctan( z )
    # end if
  # end def

# end class

## eof - $RCSfile$
