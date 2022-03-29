## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Base import *

'''
'''
class SoftMax( Base ):

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  '''
  def __call__( self, z, d = False ):
    if d:
      assert False, 'Fix this'
      return None
    else:
      e = numpy.exp( z )
      s = e.sum( )
      return e / s
    # end if
  # end def

# end class

## eof - $RCSfile$
