## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, math, numpy, sys
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
def f( X ):
  return numpy.sin( ( 4.0 * math.pi ) * X )
# end def

# --------------------------------------------------------------------------
def p( X, T ):
  if T.shape[ 1 ] > 1:
    return ( X @ T[ : , 1 : ].T ) + T[ 0 , 0 ]
  else:
    return numpy.zeros( ( X.shape[ 0 ], 1 ) ) + T[ 0 , 0 ]
  # end if
# end def

# --------------------------------------------------------------------------
def fit( X, Y, l, n ):
  R = numpy.identity( n + 1 )
  if n > 0:
    R[ 1 : , 1 : ] = ( X.T @ X ) / float( m )
    mX = X.mean( axis = 0 )
    R[ 0, 1 : ] = mX
    R[ 1 : , 0 ] = mX.T
  else:
    R = ( X.T @ X ) / float( X.shape[ 0 ] )
  # end if

  c = numpy.zeros( ( 1, n + 1 ) )
  c[ 0, 0 ] = Y.mean( )
  if n > 0:
    c[ 0, 1 : ] = numpy.multiply( X, Y ).mean( axis = 0 )
  # end if
  return c @ numpy.linalg.inv( R )
# end def

# --------------------------------------------------------------------------
def extend_polynomial( X, n ):
  for i in range( n - 1 ):
    X = \
      numpy.concatenate( \
        ( X, \
          numpy.reshape( \
              numpy.multiply( X[ : , 0 ], X[ : , i ] ), \
              ( X.shape[ 0 ], 1 ) ) \
          ), axis = 1 \
        )
  # end for
  return X
# end def

# --------------------------------------------------------------------------
# Arguments
parser = argparse.ArgumentParser( )
parser.add_argument( '-n', default = 1, type = int )
parser.add_argument( '-m', default = 10, type = int )
parser.add_argument( '-s', default = 0.05, type = float )
parser.add_argument( '-l', default = 0, type = float )
args = vars( parser.parse_args( sys.argv[ 1 : ] ) )
n = args[ 'n' ]
m = args[ 'm' ]
s = args[ 's' ]
l = args[ 'l' ]

# Test data
X = numpy.random.uniform( 0, 1, ( m, 1 ) )
Y = f( X )
X += numpy.random.normal( scale = s, size = X.shape )
Y += numpy.random.normal( scale = s, size = Y.shape )
X = extend_polynomial( X, n )

# Fit paramters
T = fit( X, Y, l, n )
print( 'Parameters: ', T )

# Draw results
draw_X = numpy.reshape( numpy.linspace( 0, 1, 100 ), ( 100, 1 ) )
draw_pX = extend_polynomial( draw_X, n )

plt.plot( draw_X, f( draw_X ), color = 'orange' )
plt.plot( draw_X, p( draw_pX, T ), color = 'red' )
plt.scatter( X[ : , 0 ], Y )
plt.show( )

## eof - $RCSfile$
