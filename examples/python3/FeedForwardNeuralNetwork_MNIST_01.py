## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import io, numpy, os, random, requests, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import PUJ.Helpers.ArgParse
import PUJ.Model.NeuralNetwork.FeedForward
import PUJ.Optimizer.Adam

## -- Parse command line arguments
parser = PUJ.Helpers.ArgParse( )
parser.add_argument( '-b', '--database', type = str )
args = parser.parse_args( )

# -- Read MNIST
if args.database is None:
  dataset_url = \
    'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
  response = requests.get( dataset_url )
  response.raise_for_status( )
  data = numpy.load( io.BytesIO( response.content ) )
else:
  data = numpy.load( args.database )
# end if
X_train = data[ 'x_train' ]
X_test = data[ 'x_test' ]
y_train = data[ 'y_train' ]
y_test = data[ 'y_test' ]

# -- Reshape data
n = X_train.shape[ 1 ] * X_train.shape[ 2 ]
X_train = numpy.reshape( X_train, ( X_train.shape[ 0 ], n ) )
X_test = numpy.reshape( X_test, ( X_test.shape[ 0 ], n ) )
y_train = numpy.reshape( y_train, ( y_train.shape[ 0 ], 1 ) )
y_test = numpy.reshape( y_test, ( y_test.shape[ 0 ], 1 ) )

# -- Categorize outputs
u = numpy.unique( y_train )
p = len( u )
r = numpy.arange( p )
Y_train = numpy.eye( len( r ) )[ y_train ].reshape( y_train.shape[ 0 ], p )
Y_test = numpy.eye( len( r ) )[ y_test ].reshape( y_test.shape[ 0 ], p )

# -- Configure neural network
nn = PUJ.Model.NeuralNetwork.FeedForward( input_size = n )
nn.AddLayer( 32, 'Logistic', theta = 'random' )
nn.AddLayer( 10, 'SoftMax', theta = 'random' )
print( nn )

# -- Configure cost
cost = PUJ.Model.NeuralNetwork.FeedForward.Cost(
    X_train, Y_train, nn, batch_size = 0
    )
cost.SetPropagationTypeToCategoricalCrossEntropy( )

## -- Prepare debug
def debug( model, J, dJ, i, show ):
  if show:
    print( i, J, dJ )
  # end if
# end def

## -- Iterative solution
PUJ.Optimizer.Adam(
  cost,
  alpha = args.learning_rate,
  beta1 = args.beta1,
  beta2 = args.beta2,
  max_iter = args.max_iterations,
  epsilon = args.epsilon,
  regularization = args.regularization,
  reg_type = args.reg_type,
  debug_step = args.debug_step,
  debug_function = debug
  )





# parser = argparse.ArgumentParser( )
# parser.add_argument( '-b', '--database', type = str )
# parser.add_argument( '-a', '--learning_rate', type = float )
# parser.add_argument( '-I', '--max_iterations', type = int )
# parser.add_argument( '-D', '--debug_step', type = int )
# parser.add_argument( '-e', '--epsilon', type = float )
# args = parser.parse_args( )

# lr = 1e-2
# I = 1000
# e = 1e-8
# d = 10

# if args.learning_rate != None: lr = args.learning_rate
# if args.max_iterations != None: I = args.max_iterations
# if args.debug_step != None: d = args.debug_step
# if args.epsilon != None: e = args.epsilon





# -- Read MNIST (hand-written digits) database

# 


# # -- Randomly test neural network
# print( '===================================' )
# i = random.randint( 0, y_test.shape[ 0 ] )
# print( 'Test example          :', i )
# print( 'Real test output      :', y_test[ i, 0 ] )
# print( 'Estimated test output :', numpy.argmax( nn( X_test[ i ] ) ) )
# print( '===================================' )

# # -- Categorize examples
# m = y_test.shape[ 0 ]
# p = nn.GetLayerOutputSize( 1 )
# eP = numpy.eye( p )
# y_test_cat = eP[ y_test , ].reshape( m, p )

# # -- Estimate all examples
# y_estim = nn( X_test )
# y_estim_cat = eP[ numpy.argmax( y_estim, axis = 1 ) , ].reshape( m, p )

# # -- Confusion matrix and accuracy
# K = y_test_cat.T @ y_estim_cat
# acc = numpy.diag( K ).sum( ) / K.sum( )
# print( '=====================================' )
# numpy.savetxt( sys.stdout, K, fmt = '% 5d' )
# print( '=====================================' )
# print( 'Accuracy = ' + str( acc * 100 ) + '%' )
# print( '=====================================' )

## eof - $RCSfile$
