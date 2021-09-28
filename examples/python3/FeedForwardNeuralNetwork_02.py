## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import os, random, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import numpy
import PUJ.Data.Algorithms, PUJ.Data.Normalize
import PUJ.Model.NeuralNetwork.FeedForward
import PUJ.Optimizer.GradientDescent

# -- Configure network
fwd_nn = PUJ.Model.NeuralNetwork.FeedForward( )
fwd_nn.LoadParameters( sys.argv[ 1 ] )

# -- Data
D = numpy.genfromtxt( sys.argv[ 2 ], delimiter = ',' )
numpy.random.shuffle( D )
X_real = D[ : , 0 : 2 ]
y_real = D[ : , 2 : 3 ]
X_real, X_min, X_off = PUJ.Data.Normalize.MinMax( X_real )

# -- Train
cost_tra = PUJ.Model.NeuralNetwork.FeedForward.Cost( X_real, y_real, fwd_nn )
cost_tra.SetPropagationTypeToBinaryCrossEntropy( )

# -- Iterative solution
tI, nI = PUJ.Optimizer.GradientDescent(
  cost_tra,
  learning_rate = 1e-4,
  init_theta = 'none',
  max_iter = 100000,
  epsilon = 1e-8,
  debug_step = 1000,
  debug_function = None
  )

print( '===================================' )
print( fwd_nn )
print( '===================================' )

## fwd_nn.SaveParameters( 'leo.nn' )

## y_estim = ( fwd_nn( X_real ) >= 0.5 ).astype( D.dtype )
## K, acc = PUJ.Data.Algorithms.Accuracy( y_real, y_estim )

## print( K )
## print( 'Accuracy: ' + str( acc * 100 ) + '%' )


## eof - $RCSfile$
