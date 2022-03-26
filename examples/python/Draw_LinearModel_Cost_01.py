## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import matplotlib.pyplot, numpy, os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )
import PUJ_ML.Model.Linear

# Read file
D = numpy.loadtxt( sys.argv[ 1 ], delimiter = ',', skiprows = 1 )
X = D[ : , -2 : -1 ]
Y = D[ : , -1 : ]

# Create model from input data
model = PUJ_ML.Model.Linear( X, Y )

# Prepare cost
cost = model.cost( X, Y )

samples = 100
minP = model.parameters( ) - 100.0
maxP = model.parameters( ) + 100.0
spaP = ( maxP - minP ) / float( samples )

print( minP.T, maxP.T )

D = numpy.zeros( ( samples, samples ) )
for i in range( samples ):
  for j in range( samples ):
    p = numpy.multiply( numpy.matrix( [ i, j ] ).astype( float ).T, spaP ) + minP
    model.setParameters( p )
    [ J, g ] = cost.evaluate( )
    D[ i , j ] = J
  # end for
# end for

matplotlib.pyplot.imshow( D )
matplotlib.pyplot.show( )

## eof - $RCSfile$
