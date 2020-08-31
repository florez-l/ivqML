## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import sys
import ActivationFunctions
from BinaryLabelingCost import *

# -- Read data
cost = BinaryLabelingCost.ReadFromFile( sys.argv[ 1 ], ActivationFunctions.LogisticSigmoid( ) )
epsilon = float( sys.argv[ 2 ] )
alpha = float( sys.argv[ 3 ] )

# -- Train weights and bias
## TODO: w0 = numpy.asmatrix( numpy.random.rand( 1, X.shape[ 1 ] ) )
## TODO: b0 = random.uniform( -0.1, 0.1 )
## TODO: w0 = [ 0, 0 ]
## TODO: b0 = 0
#w0 = [ 0.95917343, 0.81969147 ]
#b0 = 0.07589954516790992
w0 = cost.W0( "zeros" )
b0 = cost.B0( "zeros" )
[ w, b, J, n_iter ] = cost.gradient_descent( w0, b0, alpha, epsilon )
print( "-- Gradient descent  --" )
print( "Starting parameters:", w0, b0 )
print( "w =", w )
print( "b =", b )
print( "J =", J )
print( "Iterations =", n_iter )

## eof - $RCSfile$
