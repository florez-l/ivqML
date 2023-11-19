## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, math, numpy, random, sys
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Arguments
parser = argparse.ArgumentParser( )
parser.add_argument( '-m', default = 10, type = int )
parser.add_argument( '-t', default = 4, type = int )
parser.add_argument( '-x0', default = 0, type = float )
parser.add_argument( '-y0', default = 0, type = float )
parser.add_argument( '-x1', default = 0, type = float )
parser.add_argument( '-y1', default = 1, type = float )
args = vars( parser.parse_args( sys.argv[ 1 : ] ) )
m = args[ 'm' ]
t = args[ 't' ]
x0 = args[ 'x0' ]
y0 = args[ 'y0' ]
x1 = args[ 'x1' ]
y1 = args[ 'y1' ]

# Input line
p0 = numpy.matrix( [ x0, y0 ] )
p1 = numpy.matrix( [ x1, y1 ] )
v = p1 - p0
v *= 1.0 / ( ( v @ v.T ).sum( ) ** 0.5 )
n = numpy.matrix( [ -v[ 0, 1 ], v[ 0, 0 ] ] )

# Test data
X1 = numpy.random.uniform( -10, 10, ( m, 1 ) )
X2 = numpy.random.uniform( -10, 10, ( m, 1 ) )
X = numpy.concatenate( ( X1, X2 ), axis = 1 )

# Try various logistic regressions
for i in range( t ):
  s = random.uniform( -10, 10 )
  w = n.copy( ).T
  b = -( ( ( v * s ) + p0 ) @ n.T ).sum( )
  Z = 1.0 / ( 1.0 + numpy.exp( -( ( X @ w ) + b ) ) )
  T = ( Z >= 0.5 ).astype( float )
# end for

# # Draw results
# draw_X = numpy.reshape( numpy.linspace( 0, 1, 100 ), ( 100, 1 ) )
# draw_pX = extend_polynomial( draw_X, n )
# draw_pY = numpy.concatenate( ( numpy.ones( ( draw_pX.shape[ 0 ], 1 ) ), draw_pX ), axis = 1 ) @ P.T 

# plt.plot( draw_X, f( draw_X ), color = 'orange' )
# plt.plot( draw_X, draw_pY, color = 'red' )
# plt.scatter( X[ : , 0 ], Y )
# plt.show( )

## eof - $RCSfile$
