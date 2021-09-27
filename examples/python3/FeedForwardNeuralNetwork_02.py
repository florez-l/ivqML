## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import os, random, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import numpy
import PUJ.Model.NeuralNetwork.FeedForward

# -- Configure network
fwd_nn = PUJ.Model.NeuralNetwork.FeedForward( )
fwd_nn.LoadParameters( sys.argv[ 1 ] )

print( '===================================' )
print( fwd_nn )
print( '===================================' )

# -- Data
D = numpy.genfromtxt( sys.argv[ 2 ], delimiter = ',' )
numpy.random.shuffle( D )
X_real = D[ : , 0 : 2 ]
y_real = D[ : , 2 : 3 ]

# -- Train
g = fwd_nn.BackPropagate( X_real, y_real, 'bce' )

print( g )

# n = fwd_nn.GetLayerInputSize( 0 )
# print( '===================================' )
# print( fwd_nn( [ random.randint( -10, 10 ) for i in range( n ) ] ) )
# print( '===================================' )
# print( fwd_nn( numpy.random.uniform( -10, 10, ( 10, n ) ) ) )
# print( '===================================' )

## eof - $RCSfile$
