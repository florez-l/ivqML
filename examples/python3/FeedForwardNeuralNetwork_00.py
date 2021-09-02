## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import numpy
import PUJ.Model.NeuralNetwork.FeedForward

fwd_nn = PUJ.Model.NeuralNetwork.FeedForward( input_size = 2 )
fwd_nn.AddLayer( 8, PUJ.Model.NeuralNetwork.Activation.ReLU( ), 'random' )
fwd_nn.AddLayer( 4, PUJ.Model.NeuralNetwork.Activation.Tanh( ), 'random' )
fwd_nn.AddLayer( 1, PUJ.Model.NeuralNetwork.Activation.Logistic( ), 'random' )

print( '===================================' )
print( fwd_nn )
print( '===================================' )

if len( sys.argv ) > 1:
  fwd_nn.SaveParameters( sys.argv[ 1 ] )
# end if

print( '===================================' )
print( fwd_nn( 0.3, -1.2e+4 ) )
print( '===================================' )
print( fwd_nn( numpy.random.uniform( -10, 10, ( 10, 2 ) ) ) )
print( '===================================' )

## eof - $RCSfile$
