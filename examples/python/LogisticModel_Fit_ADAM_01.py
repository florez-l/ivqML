## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, os, random, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )
import PUJ_ML.Model.Logistic
import PUJ_ML.Optimizer.ADAM
import PUJ_ML.Optimizer.Debug
import ReadData

# Options
if len( sys.argv ) < 2 or len( sys.argv ) % 2 == 1:
  print( 'Usage:', sys.argv[ 0 ], 'input_data [hyperparameters]' )
  sys.exit( 1 )
# end if
fname = sys.argv[ 1 ]
params = {
  'alpha': 1e-2,
  'beta1': 0.9,
  'beta2': 0.999,
  'lambda': 0.0,
  'regularization': 'ridge',
  'max_iter': 10000,
  'debug_iter': 1000,
  'samples': 100
  }
for a in range( 2, len( sys.argv ), 2 ):
  params[ sys.argv[ a ] ] = sys.argv[ a + 1 ]
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
debugger = PUJ_ML.Optimizer.Debug.Labeling( X, Y )

# Prepare optimizer
opt = PUJ_ML.Optimizer.ADAM( cost )
opt.setLearningRate( float( params[ 'alpha' ] ) )
opt.setFirstDamping( float( params[ 'beta1' ] ) )
opt.setSecondDamping( float( params[ 'beta2' ] ) )
opt.setLambda( float( params[ 'lambda' ] ) )
if params[ 'regularization' ].lower( ) == 'ridge':
  opt.setRegularizationToRidge( )
elif params[ 'regularization' ].lower( ) == 'lasso':
  opt.setRegularizationToRidge( )
# end if
opt.setMaximumNumberOfIterations( int( params[ 'max_iter' ] ) )
opt.setNumberOfDebugIterations( int( params[ 'debug_iter' ] ) )
opt.setDebug( debugger )
opt.fit( )

# Compute accuracy
Z = model.threshold( X )
K = \
  numpy.concatenate( ( Z, 1 - Z ), axis = 1 ).T @ \
  numpy.concatenate( ( Y, 1 - Y ), axis = 1 )
acc = K.diagonal( ).sum( ) / K.sum( )

# Show results
print( '***********************' )
print( '* Model      = ' + str( model ) )
print( '* Iterations = ' + str( opt.iterations( ) ) )
print( '* Final cost = {:.4e}'.format( cost.evaluate( )[ 0 ] ) )
print( '* Accuracy   = {:.2%}'.format( acc ) )
print( '***********************' )

debugger.keep( )

## eof - $RCSfile$
