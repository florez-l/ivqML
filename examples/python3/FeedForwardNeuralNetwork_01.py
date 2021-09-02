## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import os, random, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import numpy
import PUJ.Model.NeuralNetwork.FeedForward

fwd_nn = PUJ.Model.NeuralNetwork.FeedForward( )
fwd_nn.LoadParameters( sys.argv[ 1 ] )

print( '===================================' )
print( fwd_nn )
print( '===================================' )

n = fwd_nn.GetLayerInputSize( 0 )
print( '===================================' )
print( fwd_nn( [ random.randint( -10, 10 ) for i in range( n ) ] ) )
print( '===================================' )
print( fwd_nn( numpy.random.uniform( -10, 10, ( 10, n ) ) ) )
print( '===================================' )

## eof - $RCSfile$
