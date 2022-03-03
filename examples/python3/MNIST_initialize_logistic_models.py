## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import io, numpy, os, random, requests, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )
import PUJ.Model.Logistic

# -- Read MNIST (hand-written digits) database
if len( sys.argv ) == 1:
  dataset_url = \
    'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
  response = requests.get( dataset_url )
  response.raise_for_status( )
  data = numpy.load( io.BytesIO( response.content ) )
else:
  data = numpy.load( sys.argv[ 1 ] )
# end if

# -- Show all information in npz file
print( '********************' )
print( '** Keys found in file:' )
for k in data.files:
  print( '\t', k )
# end for
print( '********************' )

# -- Get some basic information
S = data[ 'x_train' ].shape
N = S[ 1 ] * S[ 2 ]
L = sorted( set( data[ 'y_train' ] ) )

# -- Construct all regressions
buf = str( len( L ) )
for l in L:
  m = PUJ.Model.Logistic( )
  m.setParameters( [ random.uniform( -1, 1 ) for n in range( N + 1 ) ] )
  buf += '\n' + str( l ) + ' ' + str( m )
# end for

# -- Save randomized regressions
out = open( 'mnist_logistic_models.txt', 'w' )
out.write( buf )
out.close( )

## eof - $RCSfile$
