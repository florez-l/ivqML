## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Linear import *

'''
'''
class Logistic( Linear ):

  '''
  '''
  def __init__( self, w, b = None ):
    super( ).__init__( w, b )
  # end def

  '''
  '''
  def __call__( self, *cargs, threshold = True ):
    z = 1.0 / ( 1.0 + numpy.exp( -super( ).__call__( *cargs ) ) )
    if threshold:
      return ( z >= 0.5 ).astype( z.dtype )
    else:
      return z
    # end if
  # end def

# end class

## eof - $RCSfile$
