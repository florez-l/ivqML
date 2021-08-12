## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import matplotlib.pyplot, numpy
import PUJ.Model.Linear, PUJ.Regression.MSE, PUJ.Optimizer.GradientDescent

## -------------------------------------------------------------------------
number_of_samples = 20
min_value = -10
max_value =  10
polynomial = [ 20, 2.4 ]

learning_rate = 1e-4
maximum_iterations = 10000
epsilon = 1e-8
debug_step = 10
init_theta = 0

## -------------------------------------------------------------------------
matplotlib.pyplot.ion( )
figure, axes = matplotlib.pyplot.subplots( 1, 2, sharey = False )

d_axes = axes[ 0 ]
J_axes = axes[ 1 ]
aX = []
JX = []
JY = []
d_line = None
J_line = None

## -------------------------------------------------------------------------
def debug_function( J, dJ, t, i, show ):
  global figure, d_axes, J_axes, d_line, J_line, aX, JX, JY

  if show:
    model = PUJ.Model.Linear( t )
    if not isinstance( aX, ( numpy.matrix ) ):
      n = 50
      l = d_axes.get_xlim( )
      d = l[ 1 ] - l[ 0 ]
      aX = \
        numpy.matrix(
          [ ( ( d * ( i / n ) ) + l[ 0 ] ) for i in range( n + 1 ) ]
          ).T
      for i in range( 1, model.Dimensions( ) ):
        aX = numpy.append(
          aX,
          numpy.array( aX[ : , 0 ] ) * numpy.array( aX[ : , i - 1 ] ),
          axis = 1
          )
      # end for
      d_line, = d_axes.plot(
          aX[ : , 0 ], model( aX ),
          label = 'Iterative solution',
          color = 'blue', linewidth = 0.5
          )
    # end if
    d_line.set_ydata( model( aX ) )
    d_axes.legend( )

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
# Synthetic model
off_v = ( max_value - min_value ) / number_of_samples
m = PUJ.Model.Linear( polynomial )

# Synthetic data
x0 = numpy.matrix( numpy.arange( min_value, max_value + 1, off_v ) ).T
for i in range( 1, m.Dimensions( ) ):
  x0 = numpy.append(
    x0,
    numpy.array( x0[ : , 0 ] ) * numpy.array( x0[ : , i - 1 ] ),
    axis = 1
    )
# end for
y0 = m( x0 )

# Draw synthetic data
d_axes.scatter( [ x0[ : ,0 ] ], [ y0 ], color = 'red', marker = 'x' )

# Prepare regression
cost = PUJ.Regression.MSE( x0, y0 )

# Analitical solution
tA = cost.AnalyticSolve( )

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
print( '* Analitical solution  :', tA )
print( '* Iterative solution   :', tI )
print( '* Difference           :', ( ( tI - tA ) @ ( tI - tA ).T )[ 0, 0 ] ** 0.5 )
print( '* Number of iterations :', nI )
print( '=================================================================' )

## eof - $RCSfile$
