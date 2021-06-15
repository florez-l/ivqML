## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys
import Cost, GradientDescent, PGM

if len( sys.argv ) < 3:
  print( "Usage:", sys.argv[ 0 ], "input_pgm_file output_pgm_file" )
  sys.exit( 1 )
# end if

# Read an image and convert it to examples
pgm_image = PGM.Read( sys.argv[ 1 ] )
X = []
y = []
for i in range( len( pgm_image ) ):
  for j in range( len( pgm_image[ i ] ) ):
    X.append( [ float( i ), float( j ) ] )
    y.append( float( pgm_image[ i ][ j ] ) )
  # end for
# end for

cost = Cost.MaximumLikelihood( X, y )
[ W, b, nIter ] = GradientDescent.Solve(
    cost,
    learning_rate = 1e-5,
    max_iterations = 500,
    debug_step = 10
    )

print( '**********************************************' )
print( 'Gradient descent:', W, b, '(' + str( nIter ) + ')' )
print( '**********************************************' )

## eof - $RCSfile$
