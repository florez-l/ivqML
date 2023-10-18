## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, math, numpy, sys, time
import matplotlib.pyplot as plt
import Helpers, LinearRegression

'''
'''
class LinearModel:

  m_T = None

  '''
  '''
  def __init__( self, n = 0 ):
    self.m_T = numpy.zeros( ( 1, n + 1 ) )
  # end def

  '''
  '''
  def __call__( self, X, derivative = False ):
    if derivative:
      return numpy.concatenate( ( numpy.ones( ( X.shape[ 0 ], 1 ) ), X ), axis = 1 )
    else:
      return X @ self.m_T[ : , 1 : ].T + self.m_T[ 0 , 0 ]
    # end if
  # end def

  def __iadd__( self, d ):
    self.m_T += d
    return self
  # end def

  '''
  '''
  def __getitem__( self, k ):
    return self.m_T[ 0 , k ]
  # end def

  '''
  '''
  def __setitem__( self, k, v ):
    self.m_T[ 0 , k ] = v
  # end def

  '''
  '''
  def __str__( self ):
    s = str( self.m_T.size )
    for i in range( self.m_T.size ):
      s += ' ' + str( self.m_T[ 0 , i ] )
    # end for
    return s
  # end def

# end class

class MSE:

  m_Model = None
  m_X = None
  m_Y = None

  '''
  '''
  def __init__( self, model, X, Y ):
    self.m_Model = model
    self.m_X = X
    self.m_Y = Y
  # end def

  '''
  '''
  def __call__( self ):
    dif = self.m_Model( X ) - Y
    der = self.m_Model( X, True )
    return ( ( dif ** 2 ).mean( ), numpy.multiply( dif, der ).mean( axis = 0 ) * 2.0 )
  # end def

  '''
  '''
  def model( self ):
    return self.m_Model
  # end def
# end class

class GradientDescent:

  m_Cost = None
  m_A = 1e-1
  m_L = 0
  m_R = 2
  m_Debug = None

  '''
  '''
  def __init__( self, cost ):
    self.m_Cost = cost
  # end def

  '''
  '''
  def set_learning_rate( self, a ):
    self.m_A = a
  # end def
  
  '''
  '''
  def set_regularization_to_LASSO( self ):
    self.m_R = 1
  # end def

  '''
  '''
  def set_regularization_to_ridge( self ):
    self.m_R = 2
  # end def

  '''
  '''
  def set_regularization_coefficient( self, l ):
    self.m_L = l
  # end def

  '''
  '''
  def set_debug( self, d ):
    self.m_Debug = d
  # end def

  '''
  '''
  def fit( self ):
    stop = False
    m = self.m_Cost.model( )
    a = self.m_A * float( -1 )
    pD = None
    i = 0
    while not stop:
      J, D = self.m_Cost( )
      d = math.inf
      if not pD is None:
        #d = ( ( pD - D ) @ ( pD - D ).T ) ** 0.5
        d = ( D @ D.T ) ** 0.5
      # end if
      if i % 1000 == 0:
        self.m_Debug( J, d )
      # end if
      stop = d < 1e-8
      m += D * a
      pD = D
      i += 1
    # end while
  # end def

# end class

# --------------------------------------------------------------------------
# Arguments
parser = argparse.ArgumentParser( )
parser.add_argument( '-d', default = '', type = str )
parser.add_argument( '-n', default = 1, type = int )
parser.add_argument( '-s', default = 0.05, type = float )
parser.add_argument( '-a', default = 1e-1, type = float )
parser.add_argument( '-l', default = 0, type = float )
parser.add_argument( '-r', default = 2, type = int )
args = vars( parser.parse_args( sys.argv[ 1 : ] ) )
d = args[ 'd' ]
n = args[ 'n' ]
s = args[ 's' ]
a = args[ 'a' ]
l = args[ 'l' ]
r = args[ 'r' ]

# Read data
D = numpy.genfromtxt( d, delimiter = ',' )
X = D[ : , : -1 ]
Y = D[ : , -1 : ]
X = Helpers.extend_polynomial( X, n )

# Model and cost
model = LinearModel( X.shape[ 1 ] )
cost = MSE( model, X, Y )
print( 'Initial model:', model )

# Visual debugger
draw_X = numpy.reshape( numpy.linspace( 0, 1, 100 ), ( 100, 1 ) )
draw_pX = Helpers.extend_polynomial( draw_X, n )

plt.ion( )
figure, ax = plt.subplots( )
line1, = ax.plot( draw_X, model( draw_pX ), color = 'red' )
plt.scatter( X[ : , 0 ], Y )

def visual_debug( J, d ):
  # print( d )
  plt.title( 'J={:.3e}, d={:.3e}'.format( J, d ) )
  line1.set_ydata( model( draw_pX ) )
  figure.canvas.draw( )
  figure.canvas.flush_events( )
  time.sleep( 0.01 )
# end def

# Fit with optimizer
opt = GradientDescent( cost )
opt.set_learning_rate( a )
if r == 1:
  opt.set_regularization_to_LASSO( )
else:
  opt.set_regularization_to_ridge( )
# end if
opt.set_regularization_coefficient( l )
opt.set_debug( visual_debug )
opt.fit( )
print( 'Fitted model:', model )

plt.ioff( )
plt.show( )

## eof - $RCSfile$
