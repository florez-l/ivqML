## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys

# -- Create data
m = int( sys.argv[ 2 ] )
n = int( sys.argv[ 3 ] )
L = float( sys.argv[ 4 ] )
X = numpy.matrix( numpy.rint( numpy.random.rand( m, n ) * L ) )
Y = numpy.matrix( ( X[ : , 0 ] < numpy.median( X, axis = 0 )[ 0, 0 ] ) ) * 1.0

with open( sys.argv[ 1 ], "w" ) as out_file:
  h = ""
  for i in range( n ):
    h += "x{:d},".format( i + 1 )
  # end for
  h += "y\n"
  out_file.write( h )
  for j in range( X.shape[ 0 ] ):
    s = ""
    for i in range( n ):
      s += "{:f},".format( X[ j, i ] )
    # end for
    s += "{:d}\n".format( int( Y[ j, 0 ] ) )
    out_file.write( s )
  # end for
# end with

## eof - $RCSfile$
