## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, math, numpy, sys
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
def f( X ):
  return numpy.sin( ( 2.0 * math.pi ) * X ) + 4.5
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

# Fit parameters
Xi = numpy.concatenate( ( numpy.ones( ( m, 1 ) ), X ), axis = 1 )
XiTXi = ( ( Xi.T @ Xi ) / float( m ) ) + ( numpy.identity( n + 1 ) * l )
P = ( ( Y.T @ Xi ) / float( m ) ) @ numpy.linalg.inv( XiTXi )
print( 'Parameters: ', P )

# Draw results
draw_X = numpy.reshape( numpy.linspace( 0, 1, 100 ), ( 100, 1 ) )
draw_pX = extend_polynomial( draw_X, n )
draw_pY = numpy.concatenate( ( numpy.ones( ( draw_pX.shape[ 0 ], 1 ) ), draw_pX ), axis = 1 ) @ P.T 

plt.plot( draw_X, f( draw_X ), color = 'orange' )
plt.plot( draw_X, draw_pY, color = 'red' )
plt.scatter( X[ : , 0 ], Y )
plt.show( )

## eof - $RCSfile$
