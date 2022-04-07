## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

from .Linear import *
import numpy

'''
'''
class SVM( Linear ):

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  Apply a threshold with t = 0.5
  '''
  def threshold( self, X ):
    z = self.evaluate( X )
    return ( z >= 1 ).astype( z.dtype ) - ( z <= -1 ).astype( z.dtype )
  # end def

  '''
  Add a displacement to parameters
  '''
  def _evaluate( self, X ):
    return ( X @ self.m_P[ 1 : , : ] ) - self.m_P[ 0 , 0 ]
  # end def

  '''
  Cost
  '''
  class Cost( Linear.Cost ):

    '''
    Initialize an object witha zero-sized parameters vector
    '''
    def __init__( self, model, X, Y, batch_size = 0 ):
      super( ).__init__( model, X, Y, batch_size )
    # end def

    '''
    '''
    def _evaluate( self, samples, need_gradient = False ):
      X = samples[ 0 ]
      Y = samples[ 1 ]
      n = self.m_Model.numberOfParameters( )
      m = float( X.shape[ 0 ] )
      Z = numpy.multiply( self.m_Model.evaluate( X ), Y )
      I = numpy.asarray( Z < 1 ).reshape( -1 )
      J = ( 1 - Z[ I ] ).sum( ) / m
      if need_gradient:
        g = numpy.zeros( ( self.m_Model.numberOfParameters( ), 1 ) )
        g[ 0 , 0 ] = Y[ I ].sum( ) / m
        g[ 1 : , : ] = \
          numpy.multiply( X, Y )[ I , : ].sum( axis = 0 ).\
          reshape( n - 1, 1 ) / m
        return [ J, g ]
      else:
        return [ J, None ]
      # end if
    # end def

  # end class

# end class

## eof - $RCSfile$
