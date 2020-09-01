## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy, sys

# Check command line options
if len( sys.argv ) < 6:
  print(
    "Usage: " + sys.argv[ 0 ] +
    " out_fname x_min x_max m r ai=float aj=float ..."
    )
  sys.exit( 1 )
# end if

# Get command line values
out_fname = sys.argv[ 1 ]
x_min = float( sys.argv[ 2 ] )
x_max = float( sys.argv[ 3 ] )
m = int( sys.argv[ 4 ] )
r = float( sys.argv[ 5 ] )
wb = []
for arg in sys.argv[ 6 : ]:
  i = int( arg[ 1 : arg.find( '=' ) ] )
  if len( wb ) < i + 1:
    wb += [ 0 for i in range( i + 1 - len( wb ) ) ]
  # end if
  wb[ i ] = float( arg[ arg.find( '=' ) + 1 : ] )
  # end if
# end for

# Create numpy objects
n = len( wb ) - 1
w = numpy.matrix( wb[ 1 : ] )
b = wb[ 0 ]

# Create data
X = numpy.zeros( ( m, n ) )
for i in range( m ):
  v = ( ( float( i ) / float( m - 1 ) ) * ( x_max - x_min ) ) + x_min
  X[ i ] = numpy.matrix( [ [ v ** j ] for j in range( 1, n + 1 ) ] ).T
# end for
Y = ( X @ w.T ) + b

# Insert some noise
if r > 0:
  A = numpy.asmatrix( numpy.random.rand( m, 1 ) ) * 2.0 * math.pi
  R = ( numpy.asmatrix( numpy.random.rand( m, 1 ) ) * 2.0 * r ) - r
  print( R )
  #D = numpy.append( X[ :, [ 0 ] ], Y[ :, [ 0 ] ], axis = 1 )
# end if
D = numpy.append( X, Y, axis = 1 )
numpy.random.shuffle( D )

# Create CSV header
h = ""
for i in range( D.shape[ 1 ] - 1 ):
  h += "x{:d},".format( i + 1 )
# end for

# -- Save results
csv = open( out_fname, "w" )
csv.write( h + "y\n" )
for i in range( D.shape[ 0 ] ):
  row = ""
  for j in range( D.shape[ 1 ] ):
    if j > 0:
      row += ","
    # end if
    row += "{:.4f}".format( D[ i, j ] )
  # end for
  csv.write( row + "\n" )
# end for
csv.close( )

## eof - $RCSfile$
