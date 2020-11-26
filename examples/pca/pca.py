## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys
import matplotlib.pyplot as plotter
from matplotlib import cm

# Check input arguments
if len( sys.argv ) < 2:
  print( "Usage:", sys.argv[ 0 ], "input.csv" )
  sys.exit( 1 )
# end if

# Load data
print( "reading" )
X = numpy.genfromtxt( sys.argv[ 1 ], delimiter = ',' )
print( "done: ", X.shape )

# PCA
m = X.mean( axis = 0 )
c = X - m
S = ( c.T @ c ) / float( X.shape[ 0 ] )
[ v, R ] = numpy.linalg.eig( S )
idx = v.argsort( )[ : : -1 ]   
v = v[ idx ]
R = R[ : , idx ]
retention = v / v.sum( )

# Projection onto PCA
pX = ( ( X - m ) @ R )[ : , : 1 ]

# ---> Usar pX como los datos de entrada para la RN, SVM, Arbol, Bagg...

pX = numpy.append( pX, numpy.zeros( ( pX.shape[ 0 ], 1 ) ), axis = 1 )

# Backprojection from PCA
rX = ( pX @ R.T ) + m

print( "mean:\n", m )
print( "covariance:\n", S )
print( "values:\n", v )
print( "vectors:\n", R )
print( "retention:\n", retention )
print( "inverse:\n", numpy.linalg.inv( R ) )
print( "transpose:\n", R.T )
print( "inverse - transpose:\n", numpy.linalg.inv( R ) - R.T )
print( "norm e0:\n", numpy.linalg.norm( R, axis = 0 ) )
print( "norm e1:\n", numpy.linalg.norm( R, axis = 1 ) )
print( "Frobenius norm:\n", numpy.linalg.norm( R ) )

# Show initial configuration
fig, ax1 = plotter.subplots( )
ax1.axis( 'equal' )
data_X, data_Y = zip( *X )
data_pX, data_pY = zip( *pX )
data_rX, data_rY = zip( *rX )
plotter.scatter(
    data_X, data_Y, c = "#ff0000", marker = "x"
    )
plotter.scatter(
    data_pX, data_pY, c = "#00ff00", marker = "o"
    )
plotter.scatter(
    data_rX, data_rY, c = "#0000ff", marker = "+"
    )
plotter.quiver(
    [ m[ 0 ], m[ 0 ] ], [ m[ 1 ], m[ 1 ] ],
    [ R[ 0 ][ 0 ], R[ 0 ][ 1 ] ],
    [ R[ 1 ][ 0 ], R[ 1 ][ 1 ] ]
    )
plotter.show( )


# eof - $RCSfile$
