## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import io, os, random, requests, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import numpy
import PUJ.Model.NeuralNetwork.FeedForward

# -- Read MNIST (hand-written digits) database
dataset_url = \
  'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
response = requests.get( dataset_url )
response.raise_for_status( )
data = numpy.load( io.BytesIO( response.content ) )
X_test = data[ 'x_test' ]
y_test = data[ 'y_test' ]

# -- Reshape data
n = X_test.shape[ 1 ] * X_test.shape[ 2 ]
X_test = numpy.reshape( X_test, ( X_test.shape[ 0 ], n ) )
y_test = numpy.reshape( y_test, ( y_test.shape[ 0 ], 1 ) )

# -- Read neural network
fwd_nn = PUJ.Model.NeuralNetwork.FeedForward( )
fwd_nn.LoadParameters( sys.argv[ 1 ] )
print( '===================================' )
print( fwd_nn )
print( '===================================' )

# -- Randomly test neural network
print( '===================================' )
i = random.randint( 0, y_test.shape[ 0 ] )
print( 'Test example          :', i )
print( 'Real test output      :', y_test[ i, 0 ] )
print( 'Estimated test output :', numpy.argmax( fwd_nn( X_test[ i ] ) ) )
print( '===================================' )

# -- Categorize examples
m = y_test.shape[ 0 ]
p = fwd_nn.GetLayerOutputSize( 1 )
eP = numpy.eye( p )
y_test_cat = eP[ y_test , ].reshape( m, p )

# 0 : 1 0 0 0 0 0 0 0 0 0
# 1 : 0 1 0 0 0 0 0 0 0 0
# 2 : 0 0 1 0 0 0 0 0 0 0
# 3 : 0 0 0 1 0 0 0 0 0 0
# 4 : 0 0 0 0 1 0 0 0 0 0
# 5 : 0 0 0 0 0 1 0 0 0 0
# 6 : 0
# 7 : 0
# 8 : 0
# 9 : 0


# -- Estimate all examples
y_estim = fwd_nn( X_test )
y_estim_cat = eP[ numpy.argmax( y_estim, axis = 1 ) , ].reshape( m, p )

# -- Confusion matrix and accuracy
K = y_test_cat.T @ y_estim_cat
acc = numpy.diag( K ).sum( ) / K.sum( )
print( '=====================================' )
numpy.savetxt( sys.stdout, K, fmt = '% 5d' )
print( '=====================================' )
print( 'Accuracy = ' + str( acc * 100 ) + '%' )
print( '=====================================' )

## eof - $RCSfile$
