## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

from .Linear import *
import numpy

'''
'''
class Logistic( Linear ):

  '''
  '''
  m_P = None

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
    return ( z >= 0.5 ).astype( z.dtype )
  # end def

  '''
  Add a displacement to parameters
  '''
  def _evaluate( self, X ):
    return 1.0 / ( 1.0 + numpy.exp( -super( )._evaluate( X ) ) )
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
      z = self.m_Model.evaluate( X )
      J  = numpy.log( z[ Y == 1 ] + 1e-12 ).sum( )
      J += numpy.log( 1 - z[ Y == 0 ] + 1e-12 ).sum( )
      J /= -float( X.shape[ 0 ] )
      if need_gradient:
        g = numpy.zeros( ( self.m_Model.numberOfParameters( ), 1 ) )
        g[ 0 , 0 ] = z.mean( ) - Y.mean( )
        g[ 1 : , : ] = \
           (
          numpy.matrix( numpy.multiply( X, z ) ).mean( axis = 0 ) - \
          numpy.multiply( X, Y ).mean( axis = 0 )
          ).T
        return [ J, g ]
      else:
        return [ J, None ]
      # end if
    # end def

  # end class

# end class

## eof - $RCSfile$
