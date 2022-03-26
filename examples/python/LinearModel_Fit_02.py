## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================
## This example shows an implementation of a nD linear regression with
## numpy.
## =========================================================================

import numpy, sys

# Read file
D = numpy.loadtxt( sys.argv[ 1 ], delimiter = ',', skiprows = 1 )
X = D[ : , : -1 ]
Y = D[ : , -1 : ]

# Build system
m = X.shape[ 0 ]
n = X.shape[ 1 ]
A = numpy.identity( n + 1 )
A[ 1 : n + 1 , 1 : n + 1 ] = ( X.T @ X ) / float( m )
M = X.mean( axis = 0 )
A[ 0 , 1 : ] = M
A[ 1 : , 0 ] = M.T
B = numpy.zeros( ( 1, n + 1 ) )
B[ 0 , 0 ] = Y.mean( )
B[ 0 , 1 : ] = numpy.multiply( X, Y ).mean( axis = 0 )

# Solve system
w = ( B @ numpy.linalg.inv( A ) )

# Compute cost
J = numpy.power( ( ( X @ w[ : , 1 : ].T ) + w[ 0, 0 ] ) - Y, 2 ).mean( )

# Compute cost derivative
g = numpy.zeros( ( 1, n + 1 ) )
g[ : , : 1 ] = ( ( ( X @ w[ : , 1 : ].T ) + w[ 0, 0 ] ) - Y ).mean( axis = 0 )
g[ : , 1 : ] = numpy.multiply( ( ( X @ w[ : , 1 : ].T ) + w[ 0, 0 ] ) - Y, X ).mean( axis = 0 )

# Show results
print( '***********************' )
print( '* Parameters = ' + str( w ) )
print( '* Cost       = ' + str( J ) )
print( '* Derivative = ' + str( g ) )
print( '***********************' )

## eof - $RCSfile$
