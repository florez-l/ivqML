## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, numpy, sys, time
import matplotlib.pyplot as plt
import Costs, Helpers, Optimizers, Regressions

# Arguments
parser = argparse.ArgumentParser( )
parser.add_argument( '-d', default = '', type = str )
parser.add_argument( '-n', default = 1, type = int )
parser.add_argument( '-s', default = 0.05, type = float )
parser.add_argument( '-a', default = 1e-2, type = float )
parser.add_argument( '-b1', default = 0.9, type = float )
parser.add_argument( '-b2', default = 0.999, type = float )
parser.add_argument( '-l', default = 0, type = float )
parser.add_argument( '-r', default = 2, type = int )
args = vars( parser.parse_args( sys.argv[ 1 : ] ) )
d = args[ 'd' ]
n = args[ 'n' ]
s = args[ 's' ]
a = args[ 'a' ]
b1 = args[ 'b1' ]
b2 = args[ 'b2' ]
l = args[ 'l' ]
r = args[ 'r' ]

# Read data
D = numpy.genfromtxt( d, delimiter = ',' )
X = D[ : , : -1 ]
Y = D[ : , -1 : ]
X = Helpers.extend_polynomial( X, n )

# Analytical fit
a_model = Regressions.Linear( )
a_model.fit( X, Y )

# Model and cost
o_model = Regressions.Linear( X.shape[ 1 ] )
cost = Costs.MSE( o_model, X, Y )
print( 'Initial model:', o_model )

# Visual debugger
draw_X = numpy.reshape( numpy.linspace( 0, 1, 100 ), ( 100, 1 ) )
draw_pX = Helpers.extend_polynomial( draw_X, n )
plt.ion( )
figure, ax = plt.subplots( )
line1, = ax.plot( draw_X, o_model( draw_pX ), color = 'red' )
ax.plot( draw_X, a_model( draw_pX ), color = 'green' )
plt.scatter( X[ : , 0 ], Y )

def visual_debug( J, d ):
  plt.title( 'J={:.3e}, d={:.3e}'.format( J, d ) )
  line1.set_ydata( o_model( draw_pX ) )
  figure.canvas.draw( )
  figure.canvas.flush_events( )
  time.sleep( 1e-2 )
  return False
# end def

# Fit with optimizer
opt = Optimizers.ADAM( cost )
opt.set_learning_rate( a )
if r == 1:
  opt.set_regularization_to_LASSO( )
else:
  opt.set_regularization_to_ridge( )
# end if
opt.set_regularization_coefficient( l )
opt.set_debug( visual_debug )
opt.fit( )
print( 'Optimized model:', o_model )
print( 'Fitted model   :', a_model )

line1.set_ydata( o_model( draw_pX ) )
plt.ioff( )
plt.show( )

## eof - $RCSfile$
