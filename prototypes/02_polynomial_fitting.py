## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, math, numpy, sys
import matplotlib.pyplot as plt
import Helpers, LinearRegression

# --------------------------------------------------------------------------
# Arguments
parser = argparse.ArgumentParser( )
parser.add_argument( '-d', default = '', type = str )
parser.add_argument( '-n', default = 1, type = int )
parser.add_argument( '-s', default = 0.05, type = float )
parser.add_argument( '-l', default = 0, type = float )
parser.add_argument( '-r', default = 2, type = int )
args = vars( parser.parse_args( sys.argv[ 1 : ] ) )
d = args[ 'd' ]
n = args[ 'n' ]
s = args[ 's' ]
l = args[ 'l' ]
r = args[ 'r' ]

# Read data
D = numpy.genfromtxt( d, delimiter = ',' )
X = D[ : , : -1 ]
Y = D[ : , -1 : ]
X = Helpers.extend_polynomial( X, n )

# Fit parameters
T = LinearRegression.fit( X, Y, r, l, n )
print( 'Parameters: ', T )

# Draw results
draw_X = numpy.reshape( numpy.linspace( 0, 1, 100 ), ( 100, 1 ) )
draw_pX = Helpers.extend_polynomial( draw_X, n )

plt.plot( draw_X, LinearRegression.model( draw_pX, T ), color = 'red' )
plt.scatter( X[ : , 0 ], Y )
plt.show( )

## eof - $RCSfile$
