## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, cv2, math, mimetypes, numpy

# Command line options
parser = argparse.ArgumentParser( )
parser.add_argument( 'input', help = 'File where data can be found.' )
parser.add_argument( 'output', help = 'File where data can be found.' )
parser.add_argument( '-s', '--samples', type = int, default = -1 )
parser.add_argument( '-a', '--train', type = float, default = 0.7 )
args = parser.parse_args( )

# Read data
img = cv2.imread( args.input )
dsize = 8 * img.itemsize
D = numpy.argwhere( img.all( axis = -1, where = False ) ).astype( float )
Z = img.reshape( ( D.shape[ 0 ], img.shape[ -1 ] ) ).astype( float )

# Transform output data to unique labels
W = numpy.zeros( ( Z.shape[ 0 ], 1 ) )
for i in range( Z.shape[ -1 ] ):
  W += numpy.power( Z[ : , i : i + 1 ], dsize ** i )
# end for
W /= W.max( )
L = list( set( W.flatten( ) ) )
W *= float( len( L ) - 1 )
W = W.astype( numpy.uint )

# Compute sample size
L = list( set( W.flatten( ) ) )
max_samples = math.inf
for l in L:
  n = ( W[ : , 0 ] == l ).astype( int ).sum( )
  if n < max_samples:
    max_samples = n
  # end if
# end for

if args.samples > 0 and args.samples < max_samples:
  max_samples = args.samples
# end if
train_samples = int( float( max_samples ) * args.train )
test_samples = max_samples - train_samples

# Get all data
X_train = None
X_test = None
Y_train = None
Y_test = None
for l in range( len( L ) ):
  I = D[ W[ : , 0 ] == l ]
  numpy.random.shuffle( I )
  if X_train is None:
    X_train = I[ : train_samples , : ]
    X_test = I[ train_samples : train_samples + test_samples , : ]
    Y_train = numpy.ones( ( train_samples, 1 ) ) * l
    Y_test = numpy.ones( ( test_samples, 1 ) ) * l
  else:
    X_train = numpy.concatenate( ( X_train, I[ : train_samples , : ] ) )
    X_test = numpy.concatenate(
      ( X_test, I[ train_samples : train_samples + test_samples , : ] )
      )
    Y_train = numpy.concatenate(
      ( Y_train, numpy.ones( ( train_samples, 1 ) ) * l )
      )
    Y_test = numpy.concatenate(
      ( Y_test, numpy.ones( ( test_samples, 1 ) ) * l )
      )
  # end if
# end for

P = numpy.arange( X_train.shape[ 0 ] )
numpy.random.shuffle( P )
X_train = X_train[ P, : ]
Y_train = Y_train[ P, : ]

P = numpy.arange( X_test.shape[ 0 ] )
numpy.random.shuffle( P )
X_test = X_test[ P, : ]
Y_test = Y_test[ P, : ]

# Final save
ftype = mimetypes.guess_type( args.output, strict = True )[ 0 ]
if ftype == 'text/csv':
  numpy.savetxt(
    args.output, delimiter = ',', fmt = '%f',
    X = numpy.concatenate(
      ( numpy.concatenate(
        ( X_train, X_test ) ), numpy.concatenate( ( Y_train, Y_test ) )
        ),
      axis = 1
      )
    )
else:
  numpy.savez(
    args.output,
    X_train = X_train,
    Y_train = Y_train,
    X_test = X_test,
    Y_test = Y_test
    )
# end if

## eof - $RCSfile$
