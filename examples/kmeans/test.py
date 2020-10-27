## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys
import matplotlib.pyplot as plotter
import KMeans

# Check input arguments
if len( sys.argv ) < 3:
  print( "Usage:", sys.argv[ 0 ], "input.csv k" )
  sys.exit( 1 )
# end if

# Load data
X = numpy.genfromtxt( sys.argv[ 1 ], delimiter = ',' )
k = int( sys.argv[ 2 ] )
init_means = numpy.random.rand( k, X.shape[ 1 ] )

# Solve problem
#[ Y, m ] = KMeans.BruteForce( X, k )
#[ Y, m ] = KMeans.Lloyd( X, init_means )
Y = KMeans.GaussianEM( X, k )

# Show results
fig, ax1 = plotter.subplots( )
ax1.axis( 'equal' )
colors = [
    '#ff0000', '#00ff00', '#0000ff',
    '#00ffff', '#ff00ff', '#ffff00'
    ]
for i in range( k ):
  x, y = zip( *X[ numpy.where( Y == i ) ] )
  plotter.scatter( x, y, c = colors[ i % len( colors ) ], marker = "*" )
# end for
plotter.show( )

## eof - $RCSfile$
