## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, numpy, sys, time
import matplotlib.pyplot as plt
import Costs, Helpers, Optimizers, Regressions

# Arguments
parser = argparse.ArgumentParser( )
parser.add_argument( '-d', default = '', type = str )
parser.add_argument( '-a', default = 1e-2, type = float )
parser.add_argument( '-b1', default = 0.9, type = float )
parser.add_argument( '-b2', default = 0.999, type = float )
parser.add_argument( '-l', default = 0, type = float )
parser.add_argument( '-r', default = 2, type = int )
args = vars( parser.parse_args( sys.argv[ 1 : ] ) )
d = args[ 'd' ]
a = args[ 'a' ]
b1 = args[ 'b1' ]
b2 = args[ 'b2' ]
l = args[ 'l' ]
r = args[ 'r' ]

# Read data
D = numpy.genfromtxt( d, delimiter = ',' )
X = D[ : , : -1 ]
Y = D[ : , -1 : ]

Z = ( Y == 0 ).nonzero( )[ 0 ]
O = ( Y == 1 ).nonzero( )[ 0 ]
XZ = X[ Z , : ]
XO = X[ O , : ]

# Model and cost
o_model = Regressions.Logistic( X.shape[ 1 ] )
cost = Costs.CrossEntropy( o_model, X, Y )
print( 'Initial model  :', o_model )

# Visual debugger
plt.ion( )

drawN = 1000
drawX, drawY = \
  numpy.meshgrid( \
    numpy.linspace( -1, 1, drawN ), numpy.linspace( -1, 1, drawN ) \
    )
evalX = numpy.reshape( drawX, ( drawX.shape[ 0 ] * drawX.shape[ 1 ], 1 ) )
evalY = numpy.reshape( drawY, ( drawX.shape[ 0 ] * drawX.shape[ 1 ], 1 ) )
evalZ = o_model( numpy.concatenate( ( evalX, evalY ), axis = 1 ) )
drawZ = numpy.reshape( evalZ, ( drawN, drawN ) )

figure, ax = plt.subplots( )
plt.scatter( XZ[ : , 0 ], XZ[ : , 1 ] )
plt.scatter( XO[ : , 0 ], XO[ : , 1 ] )
cnt = plt.contourf( drawX, drawY, drawZ, 8, alpha = .5 )
contour_axis = plt.gca( )

def visual_debug( J, d ):
  evalZ = o_model( numpy.concatenate( ( evalX, evalY ), axis = 1 ) )
  drawZ = numpy.reshape( evalZ, ( drawN, drawN ) )
  contour_axis.clear( )
  plt.title( 'J={:.3e}, d={:.3e}'.format( J, d ) )
  plt.scatter( XZ[ : , 0 ], XZ[ : , 1 ] )
  plt.scatter( XO[ : , 0 ], XO[ : , 1 ] )
  contour_axis.contourf( drawX, drawY, drawZ, 8, alpha = .5 )
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

plt.ioff( )
plt.show( )

## eof - $RCSfile$
