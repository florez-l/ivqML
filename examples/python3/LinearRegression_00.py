## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import numpy, random
import PUJ.Model.Linear, PUJ.Regression.MSE
import PUJ.Normalize, PUJ.Optimizer.GradientDescent
import PUJ.Debug.Polynomial

## -------------------------------------------------------------------------
number_of_samples = 20
min_value = -2
max_value =  10
polynomial = [ 0, -0.21, -3 ]

learning_rate = 1e-4
regularization = 0
reg_type = 'lasso'
max_iter = 10000
epsilon = 1e-8
debug_step = 100
init_theta = \
  [ random.uniform( -1, 1 ) for i in range( len( polynomial ) ) ]

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
D = numpy.append( x0, model( x0 ), axis = 1 )
numpy.random.shuffle( D )

# Prepare data
D, D_min, D_div = PUJ.Normalize.Nothing( D )
x0 = D[ : , : -1 ]
y0 = D[ : , -1 : ]

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
  regularization = regularization,
  reg_type = reg_type,
  init_theta = init_theta,
  max_iter = max_iter,
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
