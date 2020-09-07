## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys
import ActivationFunctions
from BinaryLabelingCost import *
from Perceptron import *

# -- Create data
X = numpy.matrix( numpy.rint( numpy.random.rand( 100, 2 ) * 10 ) )
m = numpy.median( X, axis = 0 )
Y = numpy.matrix( ( X[ : , 0 ] < m[ 0, 0 ] ) ) * 1.0
e = float( sys.argv[ 2 ] )
a = float( sys.argv[ 3 ] )
l = float( sys.argv[ 4 ] )

# -- Create cost
cost = BinaryLabelingCost( X, Y, ActivationFunctions.LogisticSigmoid( ) )

# -- Train weights and bias
w0 = cost.W0( "zeros" )
b0 = cost.B0( "zeros" )
[ w, b, J, n_iter ] = cost.gradient_descent( w0, b0, a, l, e )
print( "-- Gradient descent  --" )
print( "Starting parameters:", w0, b0 )
print( "w =", w )
print( "b =", b )
print( "J =", J )
print( "Iterations =", n_iter )

p = Perceptron( w, b, ActivationFunctions.LogisticSigmoid( ) )
p.create_example_image( sys.argv[ 1 ], cost.X( ), cost.Y( ) )

## eof - $RCSfile$
