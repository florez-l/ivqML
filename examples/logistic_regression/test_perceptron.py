## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math
from Perceptron import *
import ActivationFunctions

w = [ 0.5, -0.1 ]
b = 0.3
p = Perceptron( w, b, ActivationFunctions.LogisticSigmoid( ) )

p.create_example_image( "data3.ppm", numpy.matrix( [ [ -10, -10 ], [ 10, 10 ] ] ), numpy.matrix( [ 1, 1 ] ) )

## eof - $RCSfile$
