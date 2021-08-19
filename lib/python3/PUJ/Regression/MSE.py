## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Base import *

'''
'''
class MSE( Base ):

  '''
  '''
  def __init__( self, in_X, in_y ):
    super( ).__init__( in_X, in_y )

    self.m_XtX = ( self.m_X.T / float( self.m_M ) ) @ self.m_X
    self.m_Xby = \
      ( numpy.array( self.m_X ) * numpy.array( self.m_y ) ).mean( axis = 0 )
    self.m_uX = self.m_X.mean( axis = 0 )
    self.m_uy = self.m_y.mean( )
    self.m_yty = ( self.m_y.T / float( self.m_M ) ) @ self.m_y
  # end def

  '''
  '''
  def AnalyticSolve( self ):
    x = numpy.append( numpy.array( [ self.m_uy ] ), self.m_Xby, axis = 0 )
    B = numpy.append( numpy.matrix( [ 1 ] ), self.m_uX, axis = 1 )
    A = numpy.append( self.m_uX.T, self.m_XtX, axis = 1 )
    return x @ numpy.linalg.inv( numpy.append( B, A, axis = 0 ) )
  # end def

  '''
  '''
  def CostAndGradient( self, theta ):
    b = theta[ : , 0 ]
    w = theta[ : , 1 : ]
    J = \
      ( w @ self.m_XtX @ w.T ) + \
      ( 2.0 * b * ( w @ self.m_uX.T ) ) + \
      ( b * b ) - \
      ( 2.0 * ( w @ self.m_Xby.T ) ) - \
      ( 2.0 * b * self.m_uy ) + \
      self.m_yty
    dw = 2.0 * ( ( w @ self.m_XtX ) + ( b * self.m_uX ) - self.m_Xby )
    db = 2.0 * ( ( w @ self.m_uX.T ) + b - self.m_uy )
    return [ J[ 0, 0 ], numpy.concatenate( ( db, dw ), axis = 1 ) ]
  # end def

# end class

## eof - $RCSfile$
