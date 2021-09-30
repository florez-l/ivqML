## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Linear import *

'''
'''
class Logistic( Linear ):
  '''
  '''
  def __init__( self, **kwargs ):
    super( ).__init__( **kwargs )
  # end def

  '''
  '''
  def _RealCall( self, x, **kwargs ):
    a = 1.0 / ( 1.0 + numpy.exp( -super( )._RealCall( x ) ) )
    if 'threshold' in kwargs:
      return ( a >= 0.5 ).astype( a.dtype )
    else:
      return a
    # end if
  # end def

  '''
  '''
  class Cost( Linear.Cost ):
    '''
    '''
    m_Eps = 1e-8

    '''
    '''
    def __init__( self, in_X, in_Y, model ):
      super( ).__init__( in_X, in_Y, model )
    # end def

    '''
    '''
    def Cost( self, theta ):
      self.m_Model.SetParameters( theta )
      z = self.m_Model( self.m_X )
      p = numpy.log(
        z[ numpy.where( self.m_y[ : , 0 ] == 1 )[ 0 ] , : ] + self.m_Eps
        ).sum( )
      n = numpy.log(
        1 - z[ numpy.where( self.m_y[ : , 0 ] == 0 )[ 0 ] , : ] + self.m_Eps
        ).sum( )
      return -( p + n ) / float( self.m_M )
    # end def

    '''
    '''
    def CostAndGradient( self, theta ):
      self.m_Model.SetParameters( theta )
      z = self.m_Model( self.m_X )
      p = numpy.log(
        z[ numpy.where( self.m_y[ : , 0 ] == 1 )[ 0 ] , : ] + self.m_Eps
        ).sum( )
      n = numpy.log(
        1 - z[ numpy.where( self.m_y[ : , 0 ] == 0 )[ 0 ] , : ] + self.m_Eps
        ).sum( )
      J = -( p + n ) / float( self.m_M )
      dw = numpy.matrix(
        ( numpy.array( self.m_X ) * numpy.array( z ) ).mean( axis = 0 ) -
        self.m_Xby
        )
      db = numpy.matrix( z.mean( axis = 0 ) - self.m_uy )
      return [ J, numpy.concatenate( ( db, dw ), axis = 1 ) ]
    # end def
  # end class

# end class

## eof - $RCSfile$
