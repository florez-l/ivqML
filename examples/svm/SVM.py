## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys
import matplotlib.pyplot as plotter

# Check input arguments
if len( sys.argv ) < 2:
  print( "Usage:", sys.argv[ 0 ], "input.csv" )
  sys.exit( 1 )
# end if

# Load data
D = numpy.genfromtxt( sys.argv[ 1 ], delimiter = ',' )
X = D[ : , 0 : D.shape[ 1 ] - 1 ]
Y = D[ : , D.shape[ 1 ] - 1 : ]

# Initial parameters
w = numpy.matrix( [ 1, 0 ] )
b = 0.01
alpha = 1e-2
L = 1

# Show initial configuration
fig, ax1 = plotter.subplots( )
ax1.axis( 'equal' )
dataX, dataY = zip( *X )
plotter.scatter(
  dataX, dataY, c = "#ff0000", marker = "x"
  )
plotter.quiver(
  [ 0 ], [ 0 ],
  [ -w.item( ( 0, 1 ) ) ], [ w.item( ( 0, 0 ) ) ]
  )
plotter.show( )

# Initial cost
m = float( X.shape[ 0 ] )
H = numpy.multiply( Y.T, ( w @ X.T ) - b )
mH = numpy.array( H < 1 ).flatten( ).tolist( )
Xp = X[ mH, : ]
Yp = Y[ mH, : ]
J = ( ( w @ w.T ) * L ) + ( b * b * L ) + ( ( 1 - H[ : , mH ] ).sum( ) / m )

# Main loop
stop = False
nIter = 0
while not stop:

  dw = ( 2 * L * w ) - \
       numpy.multiply( Yp, Xp ).sum( axis = 0 )
  db = ( Yp.sum( ) * b ) + ( 2 * b * L )

  w = w - ( ( alpha / m ) * dw )
  b = b - ( ( alpha / m ) * db )

  H = numpy.multiply( Y.T, ( w @ X.T ) - b )
  mH = numpy.array( H < 1 ).flatten( ).tolist( )
  Xp = X[ mH, : ]
  Yp = Y[ mH, : ]
  Jn = ( ( w @ w.T ) * L ) + ( ( 1 - H[ : , mH ] ).sum( ) / m )

  if J - Jn < 1e-6:
    stop = True
  # end if
  print( "dJ =", ( J - Jn ).item( ), "Iteration =", nIter )
  J = Jn
  nIter = nIter + 1
# end while

plotter.scatter(
  dataX, dataY, c = "#ff0000", marker = "x"
  )
plotter.quiver(
  [ 0 ], [ 0 ],
  [ -w.item( ( 0, 1 ) ) ], [ w.item( ( 0, 0 ) ) ]
  )
plotter.show( )

# eof - $RCSfile$
