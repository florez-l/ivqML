## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Base import *

'''
'''
class MaximumLikelihood( Base ):

  '''
  '''
  m_Eps = 1e-8

  '''
  '''
  def __init__( self, in_X, in_y ):
    super( ).__init__( in_X, in_y )

    self.m_Xby = \
      ( numpy.array( self.m_X ) * numpy.array( self.m_y ) ).mean( axis = 0 )
    self.m_uy = self.m_y.mean( )
  # end def

  '''
  '''
  def CostAndGradient( self, theta ):
    b = theta[ : , 0 ]
    w = theta[ : , 1 : ]
    z = 1.0 / ( 1.0 + numpy.exp( -( ( self.m_X @ w.T ) + b ) ) )
    p = numpy.log(
      z[ numpy.where( self.m_y[ : , 0 ] == 1 )[ 0 ] , : ] + self.m_Eps
      ).sum( )
    n = numpy.log(
      1 - z[ numpy.where( self.m_y[ : , 0 ] == 0 )[ 0 ] , : ] + self.m_Eps
      ).sum( )
    J = -( p + n ) / self.m_M
    dw = numpy.matrix(
      ( numpy.array( self.m_X ) * numpy.array( z ) ).mean( axis = 0 ) -
        self.m_Xby
      )
    db = numpy.matrix( z.mean( axis = 0 ) - self.m_uy )

    return [ J, numpy.concatenate( ( db, dw ), axis = 1 ) ]
  # end def

# end class

## eof - $RCSfile$
