## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, random

## -------------------------------------------------------------------------
def KMeans( X, k, initialization = 'forgy', debug = None ):
  m = X.shape[ 0 ]
  n = X.shape[ 1 ]

  # Initialization
  M = None
  L = None
  if initialization == 'forgy':
    M = X[ random.sample( range( 0, m ), k ) , : ]
  else: # initialization == 'random'
    L = numpy.asarray( random.choices( range( 0, k ), k = m ) )
  # end if
  if not debug is None:
    debug( M, L )
  # end if

  # Main loop
  for j in range( 10 ):

    # Update distances
    if not M is None:
      D = None
      for c in range( k ):
        d = ( ( ( X - M[ c , : ] ) ** 2 ).sum( axis = 1 ) ** 0.5 )
        if not D is None:
          D = numpy.append( D, d.reshape( m, 1 ), axis = 1 )
        else:
          D = d.reshape( m, 1 )
        # end if
      # end for
      L = numpy.argmin( D, axis = 1 )
    else:
      M = numpy.zeros( ( k, n ) )
    # end if

    # Update means
    for c in range( k ):
      M[ c , : ] = X[ numpy.where( L == c ) ].mean( axis = 0 )
    # end for

    if not debug is None:
      debug( M, L )
    # end if

  # end for

  return M, numpy.matrix( L ).T

# end def

## eof - $RCSfile$
