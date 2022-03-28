## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import cv2, getopt, mimetypes, numpy, os, random, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )
import PUJ_ML.Model.Logistic
import PUJ_ML.Optimizer.GradientDescent

# Options
opts, args = getopt.getopt(
  sys.argv[ 1 : ],
  'a:l:r:m:d:s:',
  [ 'alpha=', 'lambda=', 'regularization=', 'max_iter=', 'debug_iter=', 'samples=' ]
  )
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

fname = args[ 0 ]

# Read data
ftype = mimetypes.guess_type( fname, strict = True )[ 0 ]
if ftype == 'image/x-portable-bitmap':

  image = cv2.imread( fname )

  z = [ 0, 0, 0 ]
  x_ones = numpy.argwhere( ( image != z ).all( axis = 2 ) )[ : params[ 'samples' ] , : ]
  x_zeros = numpy.argwhere( ( image == z ).all( axis = 2 ) )[ : params[ 'samples' ] , : ]

  y_ones = numpy.ones( ( x_ones.shape[ 0 ], 1 ) )
  y_zeros = numpy.zeros( ( x_zeros.shape[ 0 ], 1 ) )

  X = numpy.concatenate( ( x_zeros, x_ones ) )
  Y = numpy.concatenate( ( y_zeros, y_ones ) )

  I = numpy.arange( X.shape[ 0 ] )
  numpy.random.shuffle( I )
  X = X[ I, : ]
  Y = Y[ I, : ]

elif ftype == 'text/csv':
  pass
else:
  pass # ERROR
# end if

# Create a model to keep result and its associated cost function
model = PUJ_ML.Model.Logistic( )
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
