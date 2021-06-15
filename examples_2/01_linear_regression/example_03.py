## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

# -- Load modules
import sys
sys.path.append( '../../lib/python3' )

import numpy
import matplotlib.pyplot
import PUJ_ML

## -------------------------------------------------------------------------
# -- Show data
matplotlib.pyplot.ion( )
figure, axes = matplotlib.pyplot.subplots( 1, 3, sharey = False )
data_axes = axes[ 0 ]
J_axes = axes[ 1 ]
dJ_axes = axes[ 2 ]

JX = []
JY = []
dJY = []
J_line = None
dJ_line = None

## -------------------------------------------------------------------------
def DebugFunction( J, dJ, t, i, update_data ):
  global figure, JX, JY, dJY, J_line, dJ_line

  JX += [ i ]
  JY += [ J ]
  dJY += [ dJ ]
  if J_line == None:
    J_line, = J_axes.plot( JX, JY, color = 'green' )
  # end if
  if dJ_line == None:
    dJ_line, = dJ_axes.plot( JX, dJY, color = 'blue' )
  # end if

  if update_data:
    J_line.set_xdata( JX )
    J_line.set_ydata( JY )
    J_axes.relim( )
    J_axes.autoscale_view( )

    dJ_line.set_xdata( JX )
    dJ_line.set_ydata( dJY )
    dJ_axes.relim( )
    dJ_axes.autoscale_view( )

    figure.canvas.draw( )
    figure.canvas.flush_events( )
  # end if

# end def

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
X = numpy.matrix(
  [ ( ( x1 - x0 ) * float( i ) / float( m - 1 ) ) + x0 for i in range( m ) ]
  ).T
for i in range( n - 1 ):
  X = numpy.append( X, numpy.power( X[ : ,0 ], i + 2 ), axis = 1 )
# end for
Y = ( X @ w.T ) + b
X += numpy.random.randn( m, n ) * r
Y += numpy.random.randn( m, 1 ) * r

data_axes.scatter( [ X[ : ,0 ] ], [ Y ], color = 'red', marker = '+' )

# -- Solve regression problem
J = PUJ_ML.Regression.MSECost( X, Y )
g_theta, iterations = PUJ_ML.Optimizer.GradientDescent(
  J, learning_rate = alpha,
  init_theta = 0,
  maximum_iterations = 5000,
  debug_function = DebugFunction, debug_step = 10
  )
a_theta = J.AnalyticSolve( )
diff_theta = g_theta - a_theta
diff = ( diff_theta @ diff_theta.T )[ 0, 0 ]

print( '=================================================================' )
print( 'Analytic solution    : ' + str( a_theta ) )
print( 'Gradient descent     : ' + str( g_theta ) )
print( 'Difference           : ' + str( diff ) )
print( 'Number of iterations : ' + str( iterations ) )
print( '=================================================================' )

matplotlib.pyplot.ioff( )

vX = numpy.ones( ( m, 1 ) )
vX = numpy.append(
  vX,
  numpy.matrix(
    [ ( ( x1 - x0 ) * float( i ) / float( m - 1 ) ) + x0 for i in range( m ) ]
    ).T,
  axis = 1
  )
for i in range( n - 1 ):
  vX = numpy.append( vX, numpy.power( vX[ : ,1 ], i + 2 ), axis = 1 )
# end for

g_vY = vX @ g_theta.T
a_vY = vX @ a_theta.T
data_axes.plot( vX[ : ,1 ], g_vY, color = 'green' )
data_axes.plot( vX[ : ,1 ], a_vY, color = 'blue' )

matplotlib.pyplot.show( )

## eof - $RCSfile$
