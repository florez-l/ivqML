## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import sys
import ActivationFunctions
from BinaryLabelingCost import *
from Perceptron import *

# -- Read data
if len( sys.argv ) < 6:
  print( "Usage:", sys.argv[ 0 ], " csv_file ppm_file epsilon alpha lambda" )
  sys.exit( 1 )
# end if

e = float( sys.argv[ 3 ] )
a = float( sys.argv[ 4 ] )
l = float( sys.argv[ 5 ] )

cost = BinaryLabelingCost.ReadFromFile( sys.argv[ 1 ], ActivationFunctions.LogisticSigmoid( ), e )

# -- Train weights and bias
w0 = cost.W0( "random" )
b0 = cost.B0( "random" )
print( "-- Gradient descent  --" )
print( "Starting parameters:", w0, b0 )
[ w, b, J, n_iter ] = cost.gradient_descent( w0, b0, a, l )
print( "w =", w )
print( "b =", b )
print( "J =", J )
print( "Iterations =", n_iter )

p = Perceptron( w, b, ActivationFunctions.LogisticSigmoid( ) )
p.create_example_image( sys.argv[ 2 ], cost.X( ), cost.Y( ) )
p.confussion_matrix( cost.X( ), cost.Y( ) )

## eof - $RCSfile$
