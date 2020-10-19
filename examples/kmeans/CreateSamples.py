## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy, sys
import matplotlib.pyplot as plt

radii = [ [ 5, 5 ], [ 5, 5 ], [ 3, 3 ] ]
centers = [ [ 0, 0 ], [ 12, 12 ], [ 20, 0 ] ]
angles = [ 0, 0, 0 ]
n = [ 100, 200, 150 ]

X = None
start = True
for i in range( len( radii ) ):
  Ri = numpy.random.uniform( low = 0, high = 1.5, size = ( n[ i ], 1 ) )
  Ti = numpy.random.uniform( low = 0, high = 2 * math.pi, size = ( n[ i ], 1 ) )
  Oi = numpy.ones( ( n[ i ], 1 ) )
  Xi = numpy.append( Ri * numpy.cos( Ti ), Ri * numpy.sin( Ti ), axis = 1 )
  Xi = numpy.append( Xi, Oi, axis = 1 )

  t = numpy.matrix( [ [ 1, 0, centers[ i ][ 0 ] ], [ 0, 1, centers[ i ][ 1 ] ], [ 0, 0, 1 ] ] )
  t = t * numpy.matrix( [ [ math.cos( angles[ i ] ), -math.sin( angles[ i ] ), 0 ], [ math.sin( angles[ i ] ), math.cos( angles[ i ] ), 0 ], [ 0, 0, 1 ] ] )
  t = t * numpy.matrix( [ [ radii[ i ][ 0 ], 0, 0 ], [ 0, radii[ i ][ 1 ], 0 ], [ 0, 0, 1 ] ] )
  Xi = numpy.delete( ( t * Xi.T ).T, 2, axis = 1 )
  if start:
    X = Xi
    start = False
  else:
    X = numpy.append( X, Xi, axis = 0 )
# end for

# Show data
fig, ax1 = plt.subplots( nrows = 1 )
ax1.axis( "equal" )
plt.scatter( numpy.squeeze( numpy.asarray( X[ : , 0 ] ) ), numpy.squeeze( numpy.asarray( X[ : , 1 ] ) ), c = "#ff0000", marker = "x" )
plt.show( )

if len( sys.argv ) > 1:
  numpy.savetxt( sys.argv[ 1 ], X, delimiter = "," )
# end if

## eof - $RCSfile$
