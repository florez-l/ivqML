## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math
from Perceptron import *

## -------------------------------------------------------------------------
def logistic_sigmoid( x ):
  return 1.0 / ( 1.0 + math.exp( -x ) )
# end def

w = [ 1, 1, 2, 3 ]
b = 0.3
p = Perceptron( w, b, logistic_sigmoid, 0.5 )
print( p( [ 0, 0, 1, 3 ] ) )

## eof - $RCSfile$
