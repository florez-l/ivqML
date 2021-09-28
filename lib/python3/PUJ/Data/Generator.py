## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy

'''
'''
def Ellipse( radii, center, angle, n ):
  # Unitary circle
  t = numpy.random.uniform( 0.0, 2.0 * math.pi, ( n, 1 ) )
  r = numpy.random.uniform( 0.0, 1.0, ( n, 1 ) )
  c = numpy.cos( t )
  s = numpy.sin( t )
  X = numpy.zeros( ( n, 2 ) )
  X[ : , 0 : 1 ] = ( numpy.array( r ) * numpy.array( c ) ).reshape( ( n, 1 ) )
  X[ : , 1 : 2 ] = ( numpy.array( r ) * numpy.array( s ) ).reshape( ( n, 1 ) )

  # Radii
  X[ : , 0 : 2 ] = numpy.array( X[ : , 0 : 2 ] ) * numpy.array( radii )

  # Rotation
  rc, rs = numpy.cos( angle ), numpy.sin( angle )
  R = numpy.array( ( ( rc, -rs ), ( rs, rc ) ) )
  X[ : , 0 : 2 ] = ( R @ X[ : , 0 : 2 ].T ).T

  # Translation
  X[ : , 0 : 2 ] = numpy.array( X[ : , 0 : 2 ] ) + numpy.array( center )

  return X
# end def

## eof - $RCSfile$
