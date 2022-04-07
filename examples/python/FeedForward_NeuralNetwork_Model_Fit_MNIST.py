## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, io, numpy, os, random, requests, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )
import PUJ_ML

# Command line options
parser = argparse.ArgumentParser( )
parser.add_argument( 'input_data', help = 'File/URL where data can be found.' )
parser.add_argument( '-a', type = float, default = 1e-2 )
parser.add_argument( '-b1', type = float, default = 0.9 )
parser.add_argument( '-b2', type = float, default = 0.999 )
parser.add_argument( '-l', type = float, default = 0 )
parser.add_argument( '-m', type = int, default = 10000 )
parser.add_argument( '-d', type = int, default = 1000 )
parser.add_argument( '-t', type = float, default = 0.7 )
parser.add_argument(
  '-r', type = str, default = 'ridge', help = '[ridge/LASSO]'
  )
args = parser.parse_args( )

# Read data
try:
  data = numpy.load( args.input_data )
  x_train = data[ 'x_train' ]
  y_train = data[ 'y_train' ]
  x_test = data[ 'x_test' ]
  y_test = data[ 'y_test' ]
except Exception as err:
  response = requests.get(
    'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    )
  response.raise_for_status( )
  data = numpy.load( io.BytesIO( response.content ) )
  x_train = data[ 'x_train' ]
  y_train = data[ 'y_train' ]
  x_test = data[ 'x_test' ]
  y_test = data[ 'y_test' ]
  numpy.savez(
    args.input_data,
    x_train = x_train, y_train = y_train,
    x_test = x_test, y_test = y_test
    )
# end try

# Reshape data
n = x_train.shape[ 1 ] * x_train.shape[ 2 ]
x_train = numpy.reshape( x_train, ( x_train.shape[ 0 ], n ) )
x_test = numpy.reshape( x_test, ( x_test.shape[ 0 ], n ) )

# Categorize data
L = list( set( y_test ) )
I = numpy.eye( len( L ) )
y_train = I[ y_train , : ]
y_test  = I[ y_test , : ]

# Create a model to keep result and its associated cost function
model = PUJ_ML.Model.NeuralNetwork.FeedForward( input_size = n )
model.addLayer( 'ReLU', 32 )
model.addLayer( 'SoftMax', len( L ) )
cost = model.cost( x_train[ : 100 , : ], y_train[ : 100 , : ] )

# Debugger
debugger = PUJ_ML.Optimizer.Debug.Simple( )

# Prepare optimizer
opt = PUJ_ML.Optimizer.ADAM( cost )
opt.setLearningRate( args.a )
opt.setFirstDamping( args.b1 )
opt.setSecondDamping( args.b2 )
opt.setLambda( args.l )
if args.r.lower( ) == 'ridge':
  opt.setRegularizationToRidge( )
elif args.r.lower( ) == 'lasso':
  opt.setRegularizationToLASSO( )
# end if
opt.setMaximumNumberOfIterations( args.m )
opt.setNumberOfDebugIterations( args.d )
opt.setDebug( debugger )
opt.fit( )

# Compute accuracies
# Z_train = model.threshold( X_train )
# K_train = \
#   numpy.concatenate( ( Z_train, 1 - Z_train ), axis = 1 ).T @ \
#   numpy.concatenate( ( Y_train, 1 - Y_train ), axis = 1 )
# acc_train = K_train.diagonal( ).sum( ) / K_train.sum( )

# Z_test = model.threshold( X_test )
# K_test = \
#   numpy.concatenate( ( Z_test, 1 - Z_test ), axis = 1 ).T @ \
#   numpy.concatenate( ( Y_test, 1 - Y_test ), axis = 1 )
# acc_test = K_test.diagonal( ).sum( ) / K_test.sum( )

# Show results
#print( '***********************' )
#print( '* Model          =\n' + str( model ) )
#print( '* Iterations     = ' + str( opt.iterations( ) ) )
#print( '* Final cost     = {:.4e}'.format( cost.evaluate( )[ 0 ] ) )
#print( '* Train accuracy = {:.2%}'.format( acc_train ) )
#print( '* Test accuracy  = {:.2%}'.format( acc_test ) )
#print( '***********************' )

#debugger.keep( )

## eof - $RCSfile$
