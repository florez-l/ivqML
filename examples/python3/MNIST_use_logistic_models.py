## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import io, numpy, os, random, requests, sys
import matplotlib.pyplot as plt
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )
import PUJ.Model.Logistic

# -- Read MNIST (hand-written digits) database
if len( sys.argv ) == 2:
  dataset_url = \
    'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
  response = requests.get( dataset_url )
  response.raise_for_status( )
  data = numpy.load( io.BytesIO( response.content ) )
else:
  data = numpy.load( sys.argv[ 1 ] )
# end if

# -- Read all data
X = numpy.concatenate( ( data[ 'x_train' ], data[ 'x_test' ] ), axis = 0 )
Y = numpy.concatenate( ( data[ 'y_train' ], data[ 'y_test' ] ), axis = 0 )
m = X.shape[ 0 ]

# -- Read models
models_file = open( sys.argv[ -1 ], 'r' )
models_lines = models_file.readlines( )
models_file.close( )
L = int( models_lines[ 0 ] )
labels = []
models = []
for l in models_lines[ 1 : ]:
  d = l.split( )
  labels += [ d[ 0 ] ]
  model = PUJ.Model.Logistic( )
  model.setParameters( [ float( v ) for v in d[ 2 : ] ] )
  models += [ model ]
# end for

# -- Play with the user
i = int( input( 'Type a number between 0 and ' + str( m - 1 ) + ': ' ) )
while i >= 0 and i < m:
  image = X[ i ]
  x = image.reshape( ( 1, image.shape[ 0 ] * image.shape[ 1 ] ) )
  y = []
  for model in models:
    y += [ model.evaluate( x )[ 0 , 0 ] ]
  # end for
  idx = y.index( max( y ) )
  print( '**********************' )
  print( '* Detected label :', labels[ idx ] )
  print( '* Real label     :', Y[ i ] )
  print( '**********************' )
  plt.imshow( image, cmap = 'gray' )
  plt.show( )
  i = int( input( 'Type a number between 0 and ' + str( m - 1 ) + ': ' ) )
# end while

## eof - $RCSfile$
