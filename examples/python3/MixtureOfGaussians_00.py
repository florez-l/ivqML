## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import numpy
import PUJ.Data.Generator
import PUJ.Optimizer.MixtureOfGaussians

from matplotlib import pyplot as plt

## -- Generate data
k = 3
radii = [ [ 5, 3 ], [ 1, 6 ], [ 2, 2 ] ]
centers = [ [ 0, 0 ], [ 10, 10 ], [ -10, 10 ] ]
angles = [ 0, 4.2, -3 ]
sizes = [ 100, 150, 120 ]
X = None
for i in range( len( radii ) ):
  e = PUJ.Data.Generator.Ellipse(
      radii[ i ], centers[ i ], angles[ i ], sizes[ i ]
      )
  if X is None:
    X = e
  else:
    X = numpy.append( X, e, axis = 0 );
# end for
numpy.random.shuffle( X )

# -- Debug function
# def kmeans_debug( M, L ):
#   if not L is None:
#     for i in range( k ):
#       d = X[ numpy.where( L == i ) ]
#       plt.plot( d[ : , 0 ], d[ : , 1 ], '+', label = str( i ) )
#       if not M is None:
#         plt.plot(
#           [ M[ i , : ][ 0 ] ], [ M[ i , : ][ 1 ] ],
#           'o', label = 'mean_' + str( i )
#           )
#       # end for
#     # end for
#   else:
#     plt.plot( X[ : , 0 ], X[ : , 1 ], '+', label = 'Initial data' )
#   # end if
#   plt.legend( )
#   plt.show( )
# # end def

# -- Label data
M, S, L = PUJ.Optimizer.MixtureOfGaussians( X, k )

for c in range( k ):
  print( '---------------------------' )
  print( 'Class:', c )
  print( 'Mean:' )
  print( M[ c , : ] )
  print( 'Covariance:' )
  print( S[ c ] )
  print( '---------------------------' )
# end for
        

# -- Show results
for c in range( k ):
  d = X[ numpy.where( L == c ) , : ]
  plt.plot( d[ 0 ][ : , 0 ], d[ 0 ][ : , 1 ], '+', label = str( c ) )
  plt.plot( [ M[ c , : ][ 0 ] ], [ M[ c , : ][ 1 ] ], 'o', label = 'mean_' + str( c ) )
# end for
plt.legend( )
plt.show( )

## eof - $RCSfile$
