## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, math, numpy, sys
import matplotlib.pyplot as plt
import Helpers, LinearRegression

# --------------------------------------------------------------------------
def f( X ):
  return numpy.sin( ( 2.0 * math.pi ) * X ) + 4.5
# end def

# --------------------------------------------------------------------------
# Arguments
parser = argparse.ArgumentParser( )
parser.add_argument( '-n', default = 1, type = int )
parser.add_argument( '-m', default = 10, type = int )
parser.add_argument( '-s', default = 0.05, type = float )
parser.add_argument( '-l', default = 0, type = float )
parser.add_argument( '-r', default = 2, type = int )
args = vars( parser.parse_args( sys.argv[ 1 : ] ) )
n = args[ 'n' ]
m = args[ 'm' ]
s = args[ 's' ]
l = args[ 'l' ]
r = args[ 'r' ]

# Test data
X = numpy.random.uniform( 0, 1, ( m, 1 ) )
Y = f( X )
X += numpy.random.normal( scale = s, size = X.shape )
Y += numpy.random.normal( scale = s, size = Y.shape )
X = Helpers.extend_polynomial( X, n )

# Fit parameters
T = LinearRegression.fit( X, Y, r, l, n )
print( 'Parameters: ', T )

# Draw results
draw_X = numpy.reshape( numpy.linspace( 0, 1, 100 ), ( 100, 1 ) )
draw_pX = Helpers.extend_polynomial( draw_X, n )

plt.plot( draw_X, f( draw_X ), color = 'orange' )
plt.plot( draw_X, LinearRegression.model( draw_pX, T ), color = 'red' )
plt.scatter( X[ : , 0 ], Y )
plt.show( )

## eof - $RCSfile$
