## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import io, numpy

'''
'''
class Base:

  ## -----------------------------------------------------------------------
  '''
  Paramateres for this model.
  @type numpy.matrix (nx1 column vector)
  '''
  m_P = None

  ## -----------------------------------------------------------------------
  ## Initialize an object witha zero-sized parameters vector
  ## -----------------------------------------------------------------------
  def __init__( self ):
    pass
  # end if

  ## -----------------------------------------------------------------------
  ## Parameters-related methods.
  ## -----------------------------------------------------------------------
  def parameters( self ):
    return self.m_P
  # end def

  def numberOfParameters( self ):
    if not self.m_P is None:
      return self.m_P.shape[ 0 ]
    else:
      return -1
    # end if
  # end def

  def setParameters( self, p ):
    m = numpy.matrix( p )
    self.m_P = numpy.reshape( m, ( m.size, 1 ) ).astype( float )
  # end def

  def moveParameters( self, g ):
    self.m_P += g
  # end def

  ## -----------------------------------------------------------------------
  ## Streaming methods.
  ## -----------------------------------------------------------------------
  def __str__( self ):
    b = io.BytesIO( )
    numpy.savetxt( b, self.m_P.T, fmt = '%.3f' )
    return str( self.m_P.size ) + ' ' + \
           b.getvalue( ).decode( 'latin1' )[ 0 : -1 ]
  # end def

  ## -----------------------------------------------------------------------
  ## Final-user methods.
  ## -----------------------------------------------------------------------
  def evaluate( self, x ):
    if self.m_P is None:
      raise ValueError( 'Parameters should be defined first!' )
    # end if

    rx = None
    if not isinstance( x, numpy.matrix ):
      rx = numpy.matrix( x )
    else:
      rx = x
    # end if

    if rx.shape[ 1 ] != self.m_P.shape[ 0 ] - 1:
      raise ValueError(
        'Input size (=' + str( rx.shape[ 1 ] ) +
        ') differs from parameters (=' + str( self.m_P.shape[ 0 ] - 1 )
        + ')'
        )
    # end if

    return rx
  # end def

  def threshold( self, x ):
    return self.evaluate( x )
  # end def

  ## -----------------------------------------------------------------------
  '''
  Base class for costs
  '''
  class Cost:

    ## ---------------------------------------------------------------------
    '''
    Model associated to this cost
    @type Something derived from PUJ.Model.Base
    '''
    m_Model = None
    m_X = None
    m_Y = None

    ## ---------------------------------------------------------------------
    ## Initialize an object witha zero-sized parameters vector
    ## ---------------------------------------------------------------------
    def __init__( self, model, X, Y ):
      self.m_Model = model
      self.m_X = X
      self.m_Y = Y
    # end def

    ## ---------------------------------------------------------------------
    ## Evaluate cost with gradient (if needed)
    ## ---------------------------------------------------------------------
    def evaluate( self, need_gradient = False ):
      return [ None, None ]
    # end def

    ## ---------------------------------------------------------------------
    ## Model access
    ## ---------------------------------------------------------------------
    def model( self ):
      return self.m_Model
    # end def

    ## ---------------------------------------------------------------------
    ## Move parameters
    ## ---------------------------------------------------------------------
    def updateModel( self, d ):
      self.m_Model.moveParameters( d )
    # end def
  # end class
# end class

## eof - $RCSfile$
