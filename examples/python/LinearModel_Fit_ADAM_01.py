## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================
## This example shows the fit a nD linear regression with ADAM
## =========================================================================

import getopt, numpy, os, random, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )
import PUJ_ML.Model.Linear
import PUJ_ML.Optimizer.ADAM

# Options
opts, args = getopt.getopt(
  sys.argv[ 1 : ],
  'a:b1:b2:l:r:m:d:s:',
  [ 'alpha=', 'lambda=', 'regularization=', 'max_iter=', 'debug_iter=', 'samples=' ]
  )
fname = args[ 0 ]
params = {
  'alpha': 1e-2,
  'beta1': 0.9,
  'beta2': 0.999,
  'lambda': 0.0,
  'regularization': 'ridge',
  'max_iter': 100000,
  'debug_iter': 1000,
  'samples': 100
  }
for k, v in opts:
  if k == '-a' or k == '--alpha': params[ 'alpha' ] = float( v )
  if k == '-b1' or k == '--beta1': params[ 'beta1' ] = float( v )
  if k == '-b2' or k == '--beta2': params[ 'beta2' ] = float( v )
  if k == '-l' or k == '--lambda': params[ 'lambda' ] = float( v )
  if k == '-r' or k == '--regularization': params[ 'reg' ] = v
  if k == '-m' or k == '--max_iter': params[ 'max_iter' ] = int( v )
  if k == '-d' or k == '--debug_iter': params[ 'debug_iter' ] = int( v )
  if k == '-s' or k == '--samples': params[ 'samples' ] = int( v )
# end for

# Read file
D = numpy.loadtxt( args[ 0 ], delimiter = ',', skiprows = 1 )
X = D[ : , : -1 ]
Y = D[ : , -1 : ]

# Create a model to keep result and its associated cost function
model = PUJ_ML.Model.Linear( )
model.setParameters(
  [ random.uniform( -1, 1 ) for i in range( X.shape[ 1 ] + 1 ) ]
  )
cost = model.cost( X, Y )

# Debugger
def debugger( model, i, J, dJ, show ):
  if show:
    print(
      'Iteration: {: 8d} , Cost: {:.4e} , Cost diff.: {:.4e}'.
      format( i, J, dJ )
      )
  # end if
# end def

# Prepare optimizer
opt = PUJ_ML.Optimizer.ADAM( cost )
opt.setLearningRate( params[ 'alpha' ] )
opt.setLambda( params[ 'lambda' ] )
opt.setRegularizationToRidge( )
opt.setMaximumNumberOfIterations( params[ 'max_iter' ] )
opt.setNumberOfDebugIterations( params[ 'debug_iter' ] )
opt.setDebug( debugger )
opt.fit( )

# Show results
print( '***********************' )
print( '* Model      = ' + str( model ) )
print( '* Iterations = ' + str( opt.iterations( ) ) )
print( '* Final cost = {:.4e}'.format( cost.evaluate( )[ 0 ] ) )
print( '***********************' )

## eof - $RCSfile$
