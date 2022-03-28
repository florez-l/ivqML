## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import getopt, numpy, os, random, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )
import PUJ_ML.Model.Logistic
import PUJ_ML.Optimizer.GradientDescent
import PUJ_ML.Optimizer.Debug
import ReadData

# Options
opts, args = getopt.getopt(
  sys.argv[ 1 : ],
  'a:l:r:m:d:s:',
  [ 'alpha=', 'lambda=', 'regularization=', 'max_iter=', 'debug_iter=', 'samples=' ]
  )
fname = args[ 0 ]
params = {
  'alpha': 1e-4,
  'lambda': 0.0,
  'regularization': 'ridge',
  'max_iter': 100000,
  'debug_iter': 1000,
  'samples': 100
  }
for k, v in opts:
  if k == '-a' or k == '--alpha': params[ 'alpha' ] = float( v )
  if k == '-l' or k == '--lambda': params[ 'lambda' ] = float( v )
  if k == '-r' or k == '--regularization': params[ 'reg' ] = v
  if k == '-m' or k == '--max_iter': params[ 'max_iter' ] = int( v )
  if k == '-d' or k == '--debug_iter': params[ 'debug_iter' ] = int( v )
  if k == '-s' or k == '--samples': params[ 'samples' ] = int( v )
# end for

# Read data
[ X, Y ] = ReadData.ReadData( fname )
Y = Y.mean( axis = 1 ).reshape( Y.shape[ 0 ], 1 )
Y /= Y.max( )

# Keep just some samples
if X.shape[ 0 ] > params[ 'samples' ]:

  # Extract some samples
  Z = X[ Y[ : , 0 ] == 0 ]
  O = X[ Y[ : , 0 ] == 1 ]
  numpy.random.shuffle( Z )
  numpy.random.shuffle( O )
  Z = Z[ : params[ 'samples' ] , : ]
  O = O[ : params[ 'samples' ] , : ]
  Y = numpy.concatenate(
    ( numpy.zeros( ( Z.shape[ 0 ], 1 ) ), numpy.ones( ( O.shape[ 0 ], 1 ) ) )
    )
  X = numpy.concatenate( ( Z, O ) )
# end if

# Shuffle such samples
P = numpy.arange( X.shape[ 0 ] )
numpy.random.shuffle( P )
X = X[ P , : ]
Y = Y[ P , : ]

# Create a model to keep result and its associated cost function
model = PUJ_ML.Model.Logistic( )
model.setParameters(
  [ random.uniform( -1, 1 ) for i in range( X.shape[ 1 ] + 1 ) ]
  )
cost = model.cost( X, Y )

# Debugger
debugger = PUJ_ML.Optimizer.Debug.Simple( )

# Prepare optimizer
opt = PUJ_ML.Optimizer.GradientDescent( cost )
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
