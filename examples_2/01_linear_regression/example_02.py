## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy, sys
import matplotlib.pyplot

# ## -------------------------------------------------------------------------
# def Hypothesis( w, b, x ):
#   assert isinstance( w, list ), 'w should be a list'
#   assert isinstance( x, list ), 'x should be a list'
#   assert isinstance( b, ( int, float ) ), 'b should be a number'
#   assert len( w ) == len( x ), 'w and x should have the same size'

#   n = len( w )
#   y = float( b )
#   for i in range( n ):
#     y += w[ i ] * x[ i ]
#   # end for
#   return y
# # end def

# ## -------------------------------------------------------------------------
# def Cost( w, b, X, Y ):
#   assert isinstance( w, list ), 'w should be a list'
#   assert isinstance( X, list ), 'X should be a list'
#   assert isinstance( X[ 0 ], list ), 'X should be a list of lists'
#   assert isinstance( Y, list ), 'Y should be a list'
#   assert isinstance( b, ( int, float ) ), 'b should be a number'
#   assert len( w ) == len( X[ 0 ] ), 'w and all X should have the same size'

#   m = len( X )
#   J = 0.0
#   for i in range( m ):
#     J += ( Hypothesis( w, b, X[ i ] ) - Y[ i ] ) ** 2
#   # end for
#   return J / float( m )
# # end def

# ## -------------------------------------------------------------------------
# def Gradient( w, b, X, Y ):
#   assert isinstance( w, list ), 'w should be a list'
#   assert isinstance( X, list ), 'X should be a list'
#   assert isinstance( X[ 0 ], list ), 'X should be a list of lists'
#   assert isinstance( Y, list ), 'Y should be a list'
#   assert isinstance( b, ( int, float ) ), 'b should be a number'
#   assert len( w ) == len( X[ 0 ] ), 'w and all X should have the same size'

#   m = len( X )
#   n = len( w )
#   dw = [ 0.0 for j in range( n ) ]
#   db = 0.0
#   for i in range( m ):
#     v = Hypothesis( w, b, X[ i ] ) - Y[ i ]
#     for j in range( n ):
#       dw[ j ] += v * X[ i ][ j ]
#     # end for
#     db += v
#   # end for

#   for j in range( n ):
#     dw[ j ] *= 2.0 / float( m )
#   # end for
#   db *= 2.0 / float( m )

#   return [ dw, db ]
# # end def

## -------------------------------------------------------------------------
## ---------------------------------- MAIN ---------------------------------
## -------------------------------------------------------------------------

# -- Get command line arguments
if len( sys.argv ) < 2:
  print( 'Usage: python ' + sys.argv[ 0 ] + ' wi=?? r=?? x0=?? x1=?? m=?? alpha=??' )
  sys.exit( 1 )
# end if
args = {}
args[ 'w' ] = {}
args[ 'b' ] = 0
args[ 'alpha' ] = 1.0
args[ 'r' ] = 0.0
args[ 'x0' ] = -1.0
args[ 'x1' ] = 1.0
args[ 'm' ] = 20
eps = 1e-8
for a in sys.argv:
  v = a.split( '=' )
  if len( v ) == 2:
    if v[ 0 ][ 0 ] == 'w':
      if v[ 0 ] == 'w0':
        args[ 'b' ] = v[ 1 ]
      else:
        args[ 'w' ][ int( v[ 0 ][ 1: ] ) ] = v[ 1 ]
      # end if
    else:
      args[ v[ 0 ] ] = v[ 1 ]
    # end if
  # end if
# end for

# -- Build input objects from arguments
args[ 'w' ] = sorted( args[ 'w' ].items( ), reverse = True )
w = numpy.zeros( ( 1, args[ 'w' ][ 0 ][ 0 ] ) )
for e in args[ 'w' ]:
  w[ 0, e[ 0 ] - 1 ] = float( e[ 1 ] )
# end for
b = float( args[ 'b' ] )
r = float( args[ 'r' ] )
m = int( args[ 'm' ] )
alpha = float( args[ 'alpha' ] )
x0 = float( args[ 'x0' ] )
x1 = float( args[ 'x1' ] )
eps = 1e-8
n = w.shape[ 1 ]

# -- Create data
real_X = numpy.matrix(
  [ ( ( x1 - x0 ) * float( i ) / float( m - 1 ) ) + x0 for i in range( m ) ]
  ).T
for i in range( n - 1 ):
  real_X = numpy.append(
    real_X, numpy.power( real_X[ : ,0 ], i + 2 ), axis = 1
    )
# end for
real_Y = ( real_X @ w.T ) + b
X = real_X + ( numpy.random.randn( m, n ) * r )
Y = real_Y + ( numpy.random.randn( m, 1 ) * r )

# -- Show data
matplotlib.pyplot.ion( )
figure, axes = matplotlib.pyplot.subplots( 1, 3, sharey = False )
out_axes = axes[ 0 ]
J_axes = axes[ 1 ]
dJ_axes = axes[ 2 ]
out_axes.plot( real_X[ : ,0 ], real_Y[ : ,0 ], color = 'blue' )
out_axes.scatter( [ X[ : ,0 ] ], [ Y[ : ,0 ] ], color = 'red', marker = '+' )

# -- Prepare optimization
wr = numpy.zeros( w.shape )
br = 0.0
Yr = ( X @ wr.T ) + br
J = numpy.power( Yr - Y, 2 ).mean( )
dJ = math.inf

out_line, = out_axes.plot( X[ : ,0 ], Yr[ : ,0 ], color = 'orange' )
JX = [ 0 ]
JY = [ J ]
J_line, = J_axes.plot( JX, JY, color = 'green' )

dJX = []
dJY = []
dJ_line = None

nIter = 0
while eps < dJ:

  if nIter % 50 == 0:
    figure.canvas.draw( )
    figure.canvas.flush_events( )
  # end if

  Yd = ( ( X @ wr.T ) + br ) - Y
  dw = numpy.multiply( X, Yd ).mean( axis = 0 ) * 2.0
  db = Yd.mean( ) * 2.0

  wr -= dw * alpha
  br -= db * alpha

  Yr = ( X @ wr.T ) + br
  out_line.set_ydata( Yr[ : ,0 ] )
  Jn = numpy.power( Yr - Y, 2 ).mean( )
  dJ = J - Jn
  J = Jn

  JX += [ len( JX ) ]
  JY += [ J ]
  J_line.set_xdata( JX )
  J_line.set_ydata( JY )
  J_axes.relim( )
  J_axes.autoscale_view( )

  dJX += [ len( JX ) ]
  dJY += [ dJ ]
  if dJ_line == None:
    dJ_line, = dJ_axes.plot( dJX, dJY, color = 'blue' )
  else:
    dJ_line.set_xdata( dJX )
    dJ_line.set_ydata( dJY )
    dJ_axes.relim( )
    dJ_axes.autoscale_view( )
  # end if

  nIter += 1

# end while

print( '*******************************************' )
print( 'Final w: ' + str( wr ) )
print( 'Final b: {:.3e} '.format( br ) )
print( 'Final J: {:.3e} '.format( J ) )
print( 'Final dJ: {:.3e} '.format( dJ ) )
print( 'Number of iterations: ' + str( nIter ) )
print( '*******************************************' )

## eof - $RCSfile$
