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
number_of_samples = 20
min_value = -10
max_value =  10
polynomial = [ 0, -0.21, -3 ]

learning_rate = 1e-4
maximum_iterations = 10000
epsilon = 1e-8
debug_step = 1
init_theta = 0

## -------------------------------------------------------------------------

# Synthetic model
model = PUJ.Model.Linear( polynomial )

# Synthetic data
off_value = ( max_value - min_value ) / number_of_samples
x0 = numpy.matrix( numpy.arange( min_value, max_value + 1, off_value ) ).T
for i in range( 1, model.Dimensions( ) ):
  x0 = numpy.append(
    x0,
    numpy.array( x0[ : , 0 ] ) * numpy.array( x0[ : , i - 1 ] ),
    axis = 1
    )
# end for
y0 = model( x0 )

# Shuffle data
p = numpy.random.permutation( x0.shape[ 0 ] )
x0 = x0[ p ]
y0 = y0[ p ]

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
