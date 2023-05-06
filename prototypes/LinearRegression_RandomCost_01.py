## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import csv, random, sys

# 1. Read data
X, Y = [], []
in_file = open( sys.argv[ 1 ], newline = '' )
csv_file = csv.reader( in_file, delimiter = ' ' )
for row in csv_file:
  d = [ float( v ) for v in row ]
  X += [ d[ : -1 ] ]
  Y += [ d[ -1 ] ]
# end for
in_file.close( )

# 2. Create some guesses
w = [ random.uniform( -10, 10 ) for i in range( len( X[ 0 ] ) ) ]
b = random.uniform( -10, 10 )
print( 'Guessed weights :', w )
print( 'Guessed bias    :', b )

# 3. Compute cost and its gradient
J, dJ_dw, dJ_db = 0, [ 0 for i in range( len( w ) ) ], 0
for i in range( len( X ) ):

  # 3.1. Computed results
  yp = b
  for j in range( len( X[ i ] ) ):
    yp += X[ i ][ j ] * w[ j ]
  # end for

  # 3.2. Cost
  J += ( yp - Y[ i ] ) ** 2

  # 3.3. Bias gradient
  dJ_db += ( yp - Y[ i ] )

  # 3.4. Weights gradient
  for j in range( len( X[ i ] ) ):
    dJ_dw[ j ] += ( yp - Y[ i ] ) * X[ i ][ j ]
  # end for
# end for

# 4. Averages
J /= float( len( X ) )
dJ_db *= 2.0 / float( len( X ) )
for j in range( len( X[ i ] ) ):
  dJ_dw[ j ] *= 2.0 / float( len( X ) )
# end for

print( 'Cost =', J )
print( 'Gradient =', dJ_db, dJ_dw )

## eof - $RCSfile$
