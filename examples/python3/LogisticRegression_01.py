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
learning_rate = 1e-6
maximum_iterations = 10000
epsilon = 1e-8
debug_step = 10
init_theta = [ 0.0, 0.0, 0.0 ]

## -------------------------------------------------------------------------

## Read data
D = numpy.loadtxt( open( sys.argv[ 1 ], 'rb' ), delimiter = ',' )
numpy.random.shuffle( D )

## -- Separate X and y
x0 = D[ : , : -1 ]
y0 = numpy.matrix( D[ : , -1 ] ).T

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
