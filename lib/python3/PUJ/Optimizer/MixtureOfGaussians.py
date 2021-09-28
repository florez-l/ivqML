## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy, random

## -------------------------------------------------------------------------
def MixtureOfGaussians( X, k ):

  # Initialize
  m = X.shape[ 0 ]
  n = X.shape[ 1 ]
  M = X[ random.sample( range( 0, m ), k ) , : ]
  S = [ numpy.eye( n ) for i in range( k ) ]
  P = [ 1.0 / float( k ) for i in range( k ) ]

  # Main loop
  for j in range( 1000 ):
    if j % 100 == 0:
      print( j, '/ 1000' )
    # end if

    # Compute possibilities
    r = numpy.zeros( ( m, k ) )
    for c in range( k ):
      C = X - M[ c , : ]
      S_inv = numpy.linalg.inv( S[ c ] )
      S_det = numpy.linalg.det( S[ c ] )
      S_div = ( ( 2.0 * math.pi ) ** ( n / 2.0 ) ) * ( S_det ** 0.5 )
      for i in range( m ):
        e = math.exp( -( C[ i , : ].T @ S_inv @ C[ i , : ] ) / 2.0 )
        r[ i, c ] = P[ c ] * e / S_div
      # end for
    # end for
    r = r / r.sum( axis = 1 ).reshape( m, 1 )

    # Update portions and factors
    mc = r.sum( axis = 0 )
    P = [ mc[ c ] / m for c in range( k ) ]

    # Update means and covariances
    for c in range( k ):
      M[ c , : ] = \
        ( numpy.array( r[ : , c ].reshape( m, 1 ) ) * numpy.array( X ) ).\
        sum( axis = 0 ) / mc[ c ]

      S[ c ] = numpy.zeros( ( n, n ) )
      for i in range( m ):
        C = ( X[ i , : ] - M[ c , : ] ).reshape( 1, n )
        S[ c ] += r[ i, c ] * ( C.T @ C )
      # end for
      S[ c ] /= mc[ c ]
    # end for
  # end for

  # Compute labels
  D = numpy.zeros( ( m, k ) )
  for c in range( k ):
    S_inv = numpy.linalg.inv( S[ c ] )
    for i in range( m ):
      C = ( X[ i , : ] - M[ c , : ] ).reshape( 1, n )
      D[ i, c ] = ( C @ S_inv @ C.T ) ** 0.5
    # end for
  # end for

  return M, S, numpy.matrix( numpy.argmin( D, axis = 1 ) ).T

# end def

## eof - $RCSfile$
