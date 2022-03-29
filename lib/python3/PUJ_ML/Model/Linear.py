## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

from .BaseModel import *
import io, numpy

'''
'''
class Linear( BaseModel ):

  '''
  '''
  m_P = None

  '''
  '''
  def __init__( self, X = None, Y = None ):
    if not ( X is None or Y is None ):

      # Build system
      m = X.shape[ 0 ]
      n = X.shape[ 1 ]
      A = numpy.identity( n + 1 )
      A[ 1 : n + 1 , 1 : n + 1 ] = ( X.T @ X ) / float( m )
      M = X.mean( axis = 0 )
      A[ 0 , 1 : ] = M
      A[ 1 : , 0 ] = M.T
      B = numpy.zeros( ( 1, n + 1 ) )
      B[ 0 , 0 ] = Y.mean( )
      B[ 0 , 1 : ] = numpy.multiply( X, Y ).mean( axis = 0 )

      # Solve system
      self.m_P = ( B @ numpy.linalg.inv( A ) ).T
    # end if
  # end def

  '''
  Return parameters as a numpy column vector
  '''
  def parameters( self ):
    return self.m_P
  # end def

  def numberOfParameters( self ):
    if self.m_P is None:
      return 0
    else:
      return self.m_P.shape[ 0 ]
    # end if
  # end def

  '''
  '''
  def numberOfInputs( self ):
    if self.m_P is None:
      return 0
    else:
      return self.m_P.shape[ 0 ] - 1
    # end if
  # end def

  '''
  Return the required number of inputs
  '''
  def numberOfOutputs( self ):
    return 1
  # end def

  '''
  Assign parameters
  '''
  def setParameters( self, p ):
    rP = None
    if not isinstance( p, numpy.matrix ):
      rP = numpy.matrix( p ).T
    else:
      rP = p
    # end if

    self.m_P = rP.astype( numpy.float64 )
  # end def

  '''
  Add a displacement to parameters
  '''
  def moveParameters( self, d ):
    self.m_P += d
  # end def

  '''
  Return a string-based representation of the model
  '''
  def __str__( self ):
    b = io.BytesIO( )
    numpy.savetxt( b, self.m_P.T, fmt = '%.4e' )
    return \
      str( self.m_P.size ) + ' ' + \
      b.getvalue( ).decode( 'latin1' )[ 0 : -1 ]
  # end def

  '''
  Add a displacement to parameters
  '''
  def _evaluate( self, X ):
    return ( X @ self.m_P[ 1 : , : ] ) + self.m_P[ 0 , 0 ]
  # end def

  '''
  Cost
  '''
  class Cost( BaseModel.Cost ):

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
      D = self.m_Model.evaluate( X ) - Y
      J = numpy.power( D, 2 ).mean( )
      if need_gradient:
        g = numpy.zeros( ( self.m_Model.numberOfParameters( ), 1 ) )
        g[ : 1 , : ] = D.mean( axis = 0 )
        g[ 1 : , : ] = numpy.multiply( D, X ).mean( axis = 0 ).T
        return [ J, g ]
      else:
        return [ J, None ]
      # end if
    # end def

  # end class

# end class

## eof - $RCSfile$
