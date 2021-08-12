## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import matplotlib.pyplot, numpy
import PUJ.Model.Logistic, PUJ.Regression.MaximumLikelihood
import PUJ.Optimizer.GradientDescent

## -------------------------------------------------------------------------
learning_rate = 1e-4
maximum_iterations = 10000
epsilon = 1e-8
debug_step = 100
init_theta = [ 1.0, -2.0, 3.0 ]

## -------------------------------------------------------------------------
matplotlib.pyplot.ion( )
figure, axes = matplotlib.pyplot.subplots( 1, 2, sharey = False )

d_axes = axes[ 0 ]
J_axes = axes[ 1 ]
aX = []
JX = []
JY = []
dX = None
dY = None
xD = None
d_cnt = None
J_line = None

## -------------------------------------------------------------------------
def debug_function( J, dJ, t, i, show ):
  global figure, d_axes, J_axes, d_cnt, J_line, aX, JX, JY, dX, dY, xD

  if show:
    z = PUJ.Model.Logistic( t )( xD ).reshape( dx.shape )
    if d_cnt != None:
      for coll in d_cnt.collections:
        coll.remove( )
      # end for
    # end if
    d_cnt = d_axes.contourf( dx, dy, z, alpha = 0.5 )

    JX += [ i ]
    JY += [ J ]
    if J_line == None:
      J_line, = J_axes.plot( JX, JY, color = 'green' )
    # end if

    J_line.set_label( 'Cost ({:.3e})'.format( J ) )
    J_line.set_xdata( JX )
    J_line.set_ydata( JY )
    J_axes.relim( )
    J_axes.autoscale_view( )
    J_axes.legend( )

    figure.canvas.draw( )
    figure.canvas.flush_events( )

  # end if
# end def

## -------------------------------------------------------------------------

## Read data
D = numpy.loadtxt( open( sys.argv[ 1 ], 'rb' ), delimiter = ',' )
numpy.random.shuffle( D )

## -- Separate X and y
x0 = D[ : , : -1 ]
y0 = numpy.matrix( D[ : , -1 ] ).T

x0 = x0 - ( ( x0.max( axis = 0 ) + x0.min( axis = 0 ) ) * 0.5 )

xN = x0[ numpy.where( y0[ : , 0 ] == 0 )[ 0 ] , : ]
xP = x0[ numpy.where( y0[ : , 0 ] == 1 )[ 0 ] , : ]

# Draw synthetic data
dx, dy = numpy.meshgrid(
    numpy.arange( x0[ : , 0 ].min( ) - 1, x0[ : , 0 ].max( ) + 1 ),
    numpy.arange( x0[ : , 1 ].min( ) - 1, x0[ : , 1 ].max( ) + 1 )
    )
xD = numpy.append(
  numpy.matrix( dx.ravel( ) ).T,
  numpy.matrix( dy.ravel( ) ).T,
  axis = 1
  )

d_axes.scatter(
  [ xN[ : , 0 ] ], [ xN[ : , 1 ] ],
  color = 'blue', marker = 'x', label = 'zeros'
  )
d_axes.scatter(
  [ xP[ : , 0 ] ], [ xP[ : , 1 ] ],
  color = 'orange', marker = '+', label = 'ones'
  )
d_axes.legend( )

# Prepare regression
cost = PUJ.Regression.MaximumLikelihood( x0, y0 )

# Iterative solution
tI, nI = PUJ.Optimizer.GradientDescent(
  cost,
  learning_rate = learning_rate,
  init_theta = init_theta,
  maximum_iterations = maximum_iterations,
  epsilon = epsilon,
  debug_step = debug_step,
  debug_function = debug_function
  )

print( '=================================================================' )
print( '* Iterative solution   :', tI )
print( '* Number of iterations :', nI )
print( '=================================================================' )

matplotlib.pyplot.ioff( )
matplotlib.pyplot.show( )

## eof - $RCSfile$
