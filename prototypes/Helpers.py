## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

# --------------------------------------------------------------------------
def extend_polynomial( X, n ):
  for i in range( n - 1 ):
    X = \
      numpy.concatenate( \
        ( X, \
          numpy.reshape( \
              numpy.multiply( X[ : , 0 ], X[ : , i ] ), \
              ( X.shape[ 0 ], 1 ) ) \
          ), axis = 1 \
        )
  # end for
  return X
# end def

## eof - $RCSfile$
