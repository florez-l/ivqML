## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import numpy
import PUJ.Model.Linear, PUJ.Regression.MSE
import PUJ.Optimizer.GradientDescent
import PUJ.Debug.Polynomial

## -------------------------------------------------------------------------
learning_rate = 1e-4
maximum_iterations = 1000
epsilon = 1e-8
debug_step = 10
init_theta = 0

## -------------------------------------------------------------------------

## Read data
D = numpy.loadtxt( open( sys.argv[ 1 ], 'rb' ), delimiter = ',' )
numpy.random.shuffle( D )

## -- Separate X and y
x0 = D[ : , : -1 ]
y0 = numpy.matrix( D[ : , -1 ] ).T

# Prepare regression
cost = PUJ.Regression.MSE( x0, y0 )

# Analitical solution
tA = cost.AnalyticSolve( )

# Prepare debug
debug = PUJ.Debug.Polynomial( x0[ : , 0 ], y0 )

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
print( '* Analitical solution  :', tA )
print( '* Iterative solution   :', tI )
print( '* Difference           :', ( ( tI - tA ) @ ( tI - tA ).T )[ 0, 0 ] ** 0.5 )
print( '* Number of iterations :', nI )
print( '=================================================================' )

debug.KeepFigures( )

## eof - $RCSfile$
