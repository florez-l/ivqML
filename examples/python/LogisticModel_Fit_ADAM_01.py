## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, numpy, os, random, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )
import PUJ_ML.Model.Logistic
import PUJ_ML.Optimizer.ADAM
import PUJ_ML.Optimizer.Debug

# Command line options
parser = argparse.ArgumentParser( )
parser.add_argument( 'input_data', help = 'File where data can be found.' )
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
D = numpy.loadtxt( args.input_data, delimiter = ',' )
train = int( float( D.shape[ 0 ] ) * args.t )
X_train = D[ : train , : -1 ]
Y_train = D[ : train , -1 : ]
X_test = D[ train : , : -1 ]
Y_test = D[ train : , -1 : ]

# Create a model to keep result and its associated cost function
model = PUJ_ML.Model.Logistic( )
model.setParameters(
  [ random.uniform( -1, 1 ) for i in range( X_train.shape[ 1 ] + 1 ) ]
  )
cost = model.cost( X_train, Y_train )

# Debugger
debugger = PUJ_ML.Optimizer.Debug.Labeling( X_train, Y_train )

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
Z_train = model.threshold( X_train )
K_train = \
  numpy.concatenate( ( Z_train, 1 - Z_train ), axis = 1 ).T @ \
  numpy.concatenate( ( Y_train, 1 - Y_train ), axis = 1 )
acc_train = K_train.diagonal( ).sum( ) / K_train.sum( )

Z_test = model.threshold( X_test )
K_test = \
  numpy.concatenate( ( Z_test, 1 - Z_test ), axis = 1 ).T @ \
  numpy.concatenate( ( Y_test, 1 - Y_test ), axis = 1 )
acc_test = K_test.diagonal( ).sum( ) / K_test.sum( )

# Show results
print( '***********************' )
print( '* Model          = ' + str( model ) )
print( '* Iterations     = ' + str( opt.iterations( ) ) )
print( '* Final cost     = {:.4e}'.format( cost.evaluate( )[ 0 ] ) )
print( '* Train accuracy = {:.2%}'.format( acc_train ) )
print( '* Test accuracy  = {:.2%}'.format( acc_test ) )
print( '***********************' )

debugger.keep( )

## eof - $RCSfile$
