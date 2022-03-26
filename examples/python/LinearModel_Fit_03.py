## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================
## This example shows the use of a nD linear regression
## =========================================================================

import numpy, os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )
import PUJ_ML.Model.Linear

# Read file
D = numpy.loadtxt( sys.argv[ 1 ], delimiter = ',', skiprows = 1 )
X = D[ : , : -1 ]
Y = D[ : , -1 : ]

# Create model from input data
model = PUJ_ML.Model.Linear( X, Y )

# Compute cost
[ J, g ] = model.cost( X, Y ).evaluate( need_gradient = True )

# Show results
print( '***********************' )
print( '* Model      = ' + str( model ) )
print( '* Cost       = ' + str( J ) )
print( '* Derivative = ' + str( g.T ) )
print( '***********************' )

## eof - $RCSfile$
