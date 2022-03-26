## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================
## This example shows a naive implementation of a 1D linear regression.
## =========================================================================

import csv, sys

# Read file
csv_file = open( sys.argv[ 1 ] )
csv_reader = csv.reader( csv_file, delimiter = ',' )
X = []
Y = []
for row in csv_reader:
  try:
    data = [ float( r ) for r in row ]
    X += [ data[  0 ] ]
    Y += [ data[ -1 ] ]
  except Exception as err:
    pass
  # end try
# end for
csv_file.close( )

# Build system
sx1 = float( 0 )
sx2 = float( 0 )
sxy = float( 0 )
sy = float( 0 )
for i in range( len( X ) ):
  sx1 += X[ i ]
  sx2 += X[ i ] * X[ i ]
  sxy += X[ i ] * Y[ i ]
  sy += Y[ i ]
# end for
A = [ [ 0 for i in range( 2 ) ] for j in range( 2 ) ]
A[ 0 ][ 0 ] = sx2
A[ 1 ][ 0 ] = A[ 0 ][ 1 ] = sx1
A[ 1 ][ 1 ] = float( len( X ) )
B = [ sxy, sy ]

# Solve system
dA = ( A[ 0 ][ 0 ] * A[ 1 ][ 1 ] ) - ( A[ 0 ][ 1 ] * A[ 1 ][ 0 ] )
if dA == 0:
  print( 'Error' )
  sys.exit( 1 )
# end if
iA = [ [ 0 for i in range( 2 ) ] for j in range( 2 ) ]
iA[ 0 ][ 0 ] =  A[ 1 ][ 1 ] / dA
iA[ 1 ][ 1 ] =  A[ 0 ][ 0 ] / dA
iA[ 0 ][ 1 ] = -A[ 1 ][ 0 ] / dA
iA[ 1 ][ 0 ] = -A[ 0 ][ 1 ] / dA

w = ( iA[ 0 ][ 0 ] * B[ 0 ] ) + ( iA[ 0 ][ 1 ] * B[ 1 ] )
b = ( iA[ 1 ][ 0 ] * B[ 0 ] ) + ( iA[ 1 ][ 1 ] * B[ 1 ] )

# Compute cost
J = float( 0 )
for i in range( len( X ) ):
  J += ( ( w * X[ i ] ) + b - Y[ i ] ) ** 2
# end for
J /= float( len( X ) )

# Compute cost derivative
g = [ float( 0 ), float( 0 ) ]
for i in range( len( X ) ):
  g[ 0 ] += 2 * ( ( w * X[ i ] ) + b - Y[ i ] ) * X[ i ]
  g[ 1 ] += 2 * ( ( w * X[ i ] ) + b - Y[ i ] )
# end for
g[ 0 ] /= float( len( X ) )
g[ 1 ] /= float( len( X ) )

# Show results
print( '***********************' )
print( '* Weight     = ' + str( w ) )
print( '* Bias       = ' + str( b ) )
print( '* Cost       = ' + str( J ) )
print( '* Derivative = ' + str( g ) )
print( '***********************' )

## eof - $RCSfile$
