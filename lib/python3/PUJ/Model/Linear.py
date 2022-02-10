## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

from .Base import *

'''
'''
class Linear( Base ):

  ## -----------------------------------------------------------------------
  ## Initialize an object witha zero-sized parameters vector
  ## -----------------------------------------------------------------------
  def __init__( self, data = None ):
    if not data is None:
      if not isinstance( data, numpy.matrix ):
        raise ValueError(
          'Input data is not a numpy.matrix (' + str( type( data ) ) + ')'
          )
      # end if

      # Get input matrices
      m = data.shape[ 0 ]
      n = data.shape[ 1 ] - 1
      X = data[ : , 0 : n ]
      y = data[ : , n : ]

      # Build vector
      b = numpy.zeros( ( n + 1, 1 ) )
      b[ 0 , 0 ] = y.mean( )
      b[ 1 : , : ] = numpy.multiply( X, y ).mean( axis = 0 ).T

      # Build matrix
      A = numpy.identity( n + 1 )
      A[ 1 : , 1 : ] = ( X.T @ X ) / float( m )
      A[ 0 : 1 , 1 : ] = X.mean( axis = 0 )
      A[ 1 : , 0 : 1 ] = A[ 0 : 1 , 1 : ].T

      # Solve system
      self.m_P = numpy.linalg.inv( A ).T @ b
    # end if
  # end if

  ## -----------------------------------------------------------------------
  ## Final-user methods.
  ## -----------------------------------------------------------------------
  def evaluate( self, x ):
    rx = super( ).evaluate( x )
    return ( rx @ self.m_P[ 1 : , : ] ) + self.m_P[ 0 , 0 ]
  # end def

  ## -----------------------------------------------------------------------
  '''
  MSE-based cost function for linear regressions.
  '''
  class Cost( Base.Cost ):

    ## ---------------------------------------------------------------------
    ## Initialize an object witha zero-sized parameters vector
    ## ---------------------------------------------------------------------
    def __init__( self, model, X, y ):
      super( ).__init__( model, X, y )
      self.m_XtX = ( X.T @ X ) / float( X.shape[ 0 ] )
      self.m_mX = numpy.matrix( X.mean( axis = 0 ) )
      self.m_mY = y.mean( )
      self.m_XhY = numpy.matrix( numpy.multiply( X, y ).mean( axis = 0 ) ).T
    # end def

    ## ---------------------------------------------------------------------
    ## Evaluate cost with gradient (if needed)
    ## ---------------------------------------------------------------------
    def evaluate( self, need_gradient = False ):
      J = \
        numpy.power( self.m_Model.evaluate( self.m_X ) - self.m_Y, 2 ).\
        mean( )
      if need_gradient:
        g = numpy.zeros( self.m_Model.parameters( ).shape )
        w = g[ 1 : , : ]
        b = g[ 0 , 0 ]

        g[ 0 , 0 ] = ( self.m_mX @ w ) + b - self.m_mY
        g[ 1 : , : ]  = self.m_XtX @ w
        g[ 1 : , : ] += b * self.m_mX.T
        g[ 1 : , : ] -= self.m_XhY
        
        return [ J, 2.0 * g ]
      else:
        return [ J, None ]
      # end if
    # end def
  # end class
# end class

## eof - $RCSfile$
