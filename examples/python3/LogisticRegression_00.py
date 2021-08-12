## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import matplotlib.pyplot, numpy
import PUJ.Model.Logistic, PUJ.Regression.MaximumLikelihood
import PUJ.Optimizer.GradientDescent
import PUJ.Debug.Labeling

## -------------------------------------------------------------------------
number_of_samples = 20
min_value = -10
max_value =  10

learning_rate = 1e-1
maximum_iterations = 10000
epsilon = 1e-8
debug_step = 100
init_theta = [ 0.0, 1.0, 1.0 ]

## -------------------------------------------------------------------------

# Synthetic data
off_value = max_value - min_value
x0 = ( numpy.random.rand( number_of_samples, 2 ) * off_value ) + min_value
y0 = numpy.matrix( ( x0[ : , 0 ] < 0 ).astype( x0.dtype ) ).T
xN = x0[ numpy.where( y0[ : , 0 ] == 0 )[ 0 ] , : ]
xP = x0[ numpy.where( y0[ : , 0 ] == 1 )[ 0 ] , : ]
min_samples = min( xN.shape[ 0 ], xP.shape[ 0 ] )
numpy.random.shuffle( xN )
numpy.random.shuffle( xP )
xN = xN[ : min_samples , : ]
xP = xP[ : min_samples , : ]
xN = numpy.append( xN, numpy.zeros( ( min_samples, 1 ) ), axis = 1 )
xP = numpy.append( xP, numpy.ones( ( min_samples, 1 ) ), axis = 1 )
D = numpy.append( xN, xP, axis = 0 )
numpy.random.shuffle( D )
x0 = D[ : , : 2 ]
y0 = D[ : , -1 : ]

# Prepare regression
cost = PUJ.Regression.MaximumLikelihood( x0, y0 )

# Prepare debug
debug = PUJ.Debug.Labeling( x0[ : , 0 : 2 ], y0 )

# Iterative solution
tI, nI = PUJ.Optimizer.GradientDescent(
  cost,
  learning_rate = learning_rate,
  init_theta = init_theta,
  maximum_iterations = maximum_iterations,
  epsilon = epsilon,
  debug_step = debug_step,
  debug_function = debug
  )

print( '=================================================================' )
print( '* Iterative solution   :', tI )
print( '* Number of iterations :', nI )
print( '=================================================================' )

debug.KeepFigures( )

## eof - $RCSfile$
