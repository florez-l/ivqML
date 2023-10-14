## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

# --------------------------------------------------------------------------
def model( X, T ):
  if T.shape[ 1 ] > 1:
    return ( X @ T[ : , 1 : ].T ) + T[ 0 , 0 ]
  else:
    return numpy.zeros( ( X.shape[ 0 ], 1 ) ) + T[ 0 , 0 ]
  # end if
# end def

# --------------------------------------------------------------------------
def fit( X, Y, r, l, n ):
  R = numpy.identity( n + 1 )
  if n > 0:
    R[ 1 : , 1 : ] = ( X.T @ X ) / float( X.shape[ 0 ] )
    mX = X.mean( axis = 0 )
    R[ 0, 1 : ] = mX
    R[ 1 : , 0 ] = mX.T
  else:
    R = ( X.T @ X ) / float( X.shape[ 0 ] )
  # end if
  if l != 0:
    if r == 2: # ridge
      L = numpy.identity( n + 1 ) * l
      L[ 0 , 0 ] = 0
      R += L
    else: # LASSO: does it have any meaning in analytical regression?
      # L = numpy.zeros( ( n + 1, n + 1 ) )
      # R += L
      pass
    # end if
  # end if

  c = numpy.zeros( ( 1, n + 1 ) )
  c[ 0, 0 ] = Y.mean( )
  if n > 0:
    c[ 0, 1 : ] = numpy.multiply( X, Y ).mean( axis = 0 )
  # end if
  return c @ numpy.linalg.inv( R )
# end def

## eof - $RCSfile$
