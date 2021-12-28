## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import io, numpy, random
from .Base import *

'''
'''
class Linear( Base ):

  '''
  '''
  m_Weights   = None
  m_Bias      = None

  '''
  '''
  def __init__( self, **kwargs ):
    super( ).__init__( **kwargs )
    if 'parameters' in kwargs:
      if isinstance( kwargs[ 'parameters' ], ( str ) ):
        if 'size' in kwargs:
          n = int( kwargs[ 'size' ] )
          if kwargs[ 'parameters' ] == 'zeros':
            self.SetParamters( [ float( 0 ) for i in range( n + 1 ) ] )
          elif kwargs[ 'parameters' ] == 'ones':
            self.SetParamters( [ float( 1 ) for i in range( n + 1 ) ] )
          elif kwargs[ 'parameters' ] == 'random':
            self.SetParameters(
              [ random.uniform( -1, 1 ) for i in range( n + 1 ) ]
              )
          # end if
        # end if
      else:
        self.SetParameters( kwargs[ 'parameters' ] )
      # end if
    else:
      if 'weights' in kwargs:
        self.m_Weights = numpy.matrix( kwargs[ 'weights' ] ).flatten( )
        self.m_Bias = float( 0 )
        if 'bias' in kwargs:
          self.m_Bias = kwargs[ 'bias' ]
        # end if
    # end if
  # end def

  '''
  '''
  def GetInputSize( self ):
    if not self.m_Weights is None:
      return self.m_Weights.shape[ 0 ]
    else:
      return 0
    # end if
  # end def

  '''
  '''
  def GetOutputSize( self ):
    return 1
  # end def

  '''
  '''
  def GetParameters( self ):
    T = numpy.zeros( ( 1, self.GetInputSize( ) + 1 ) )
    if not self.m_Bias is None:
      T[ 0 , 0 ] = self.m_Bias
    # end if
    if not self.m_Weights is None:
      T[ : , 1 : ] = self.m_Weights.T
    # end if
    return T
  # end def

  '''
  '''
  def SetParameters( self, T ):
    assert isinstance( T, self.m_ValidTypes ), \
           'Invalid parameters type (' + str( type( T ) ) + ')'

    mT = numpy.matrix( T ).flatten( )
    self.m_Weights = mT[ : , 1 : ].T
    self.m_Bias = mT[ 0, 0 ]
  # end def

  '''
  '''
  def Fit( self, X, y ):
    uX = numpy.matrix( X.mean( axis = 0 ) )
    x = numpy.append(
      numpy.array( [ y.mean( ) ] ),
      ( numpy.array( X ) * numpy.array( y ) ).mean( axis = 0 ),
      axis = 0
      )
    B = numpy.append( numpy.matrix( [ 1 ] ), uX, axis = 1 )
    A = numpy.append( uX.T, ( X.T / float( X.shape[ 0 ] ) ) @ X, axis = 1 )
    self.SetParameters(
      x @ numpy.linalg.inv( numpy.append( B, A, axis = 0 ) )
      )
  # end def

  '''
  '''
  def _RealCall( self, x, **kwargs ):
    return ( x @ self.m_Weights ) + self.m_Bias
  # end def

  '''
  '''
  def __str__( self ):
    r = ''
    if not self.m_Bias is None:
      r += str( self.m_Bias ) + ' '
    else:
      r += '0 '
    # end if
    if not self.m_Weights is None:
      i = io.BytesIO( )
      numpy.savetxt( i, self.m_Weights, newline = ' ' )
      r += i.getvalue( ).decode( 'ascii' )
    # end if
    return r
  # end def

  '''
  '''
  class Cost:
    '''
    '''
    m_Eps = 1e-8

    '''
    '''
    def __init__( self, in_X, in_Y, model ):
      self.m_Model = model

      self.m_X = in_X
      self.m_y = in_Y
      self.m_M = self.m_X.shape[ 0 ]

      self.m_XtX = ( self.m_X.T / float( self.m_M ) ) @ self.m_X
      self.m_Xby = \
        ( numpy.array( self.m_X ) * numpy.array( self.m_y ) ).mean( axis = 0 )
      self.m_uX = self.m_X.mean( axis = 0 )
      self.m_uy = self.m_y.mean( )
      self.m_yty = ( self.m_y.T / float( self.m_M ) ) @ self.m_y
    # end def

    '''
    '''
    def GetInitialParameters( self ):
      return self.m_Model.GetParameters( )
    # end def

    '''
    '''
    def GetModel( self ):
      return self.m_Model
    # end def

    '''
    '''
    def Cost( self, theta ):
      self.m_Model.SetParameters( theta )
      b = theta[ 0 , 0 ]
      w = theta[ : , 1 : ]
      J = \
        ( w @ self.m_XtX @ w.T ) + \
        ( 2.0 * b * ( w @ self.m_uX.T ) ) + \
        ( b * b ) - \
        ( 2.0 * ( w @ self.m_Xby.T ) ) - \
        ( 2.0 * b * self.m_uy ) + \
        self.m_yty
      return J[ 0, 0 ]
    # end def

    '''
    '''
    def CostAndGradient( self, theta ):
      J = self.Cost( theta )
      b = theta[ : , 0 ]
      w = theta[ : , 1 : ]
      db = numpy.matrix( ( w @ self.m_uX.T ) + b - self.m_uy )
      dw = ( w @ self.m_XtX ) + ( b * self.m_uX ) - self.m_Xby
      return [ J, 2.0 * numpy.concatenate( ( db, dw ), axis = 1 ) ]
    # end def
  # end class

# end class

## eof - $RCSfile$
