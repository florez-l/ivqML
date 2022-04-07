## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

from ..BaseModel import *
import io
import PUJ_ML.Model.NeuralNetwork.Activation

'''
'''
class FeedForward( BaseModel ):
  m_W = []
  m_B = []
  m_S = []
  m_InputSize = 0

  '''
  '''
  def __init__( self, input_size ):
    assert input_size > 0, 'Input size should be > 0'
    self.m_InputSize = input_size
  # end def

  '''
  '''
  def addLayer( self, a, size, init = [ -1e-1, 1e-1 ] ):

    # Input values
    o_size = size

    # Weights
    i_size = self.m_InputSize
    if len( self.m_W ) > 0:
      i_size = self.m_W[ -1 ].shape[ 0 ]
    # end if
    self.m_W += \
      [ numpy.random.uniform( init[ 0 ], init[ 1 ], ( o_size, i_size ) ) ]

    # Biases
    self.m_B += \
      [ numpy.random.uniform( init[ 0 ], init[ 1 ], ( o_size, 1 ) ) ]

    # Activation function
    if isinstance( a, ( str ) ):
      self.m_S += [ getattr( PUJ_ML.Model.NeuralNetwork.Activation, a )( ) ]
    else:
      self.m_S += [ a ]
    # end if
  # end def

  '''
  '''
  def numberOfLayers( self ):
    return len( self.m_W )
  # end def

  '''
  '''
  def layerInputSize( self, i ):
    if i >= 0 and i < len( self.m_W ):
      return self.m_W.shape[ 1 ]
    else:
      return 0
    # end if
  # end def

  '''
  '''
  def layerOutputSize( self, i ):
    if i >= 0 and i < len( self.m_W ):
      return self.m_W.shape[ 0 ]
    else:
      return 0
    # end if
  # end def

  '''
  '''
  def layerWeights( self, i ):
    if i < len( self.m_W ):
      return self.m_W[ i ]
    else:
      return None
    # end if
  # end def

  '''
  '''
  def layerBiases( self, i ):
    if i < len( self.m_B ):
      return self.m_B[ i ]
    else:
      return None
    # end if
  # end def

  '''
  '''
  def layerActivation( self, i ):
    if i < len( self.m_S ):
      return self.m_S[ i ]
    else:
      return None
    # end if
  # end def

  '''
  Return parameters as a numpy column vector
  '''
  def parameters( self ):
    p = None
    L = len( self.m_W )
    if L > 0:
      t = []
      for i in range( L ):
        t += [ self.m_W[ i ].flatten( ) ]
        t += [ self.m_B[ i ].flatten( ) ]
      # end for
      p = numpy.concatenate( tuple( t ) )
    # end if
    return p
  # end def

  '''
  '''
  def numberOfParameters( self ):
    n = 0
    for i in range( len( self.m_W ) ):
      n += ( self.m_W[ i ].shape[ 1 ] + 1 ) * self.m_W[ i ].shape[ 0 ]
    # end for
    return n
  # end def

  '''
  Return the required number of inputs
  '''
  def numberOfInputs( self ):
    return self.m_InputSize
  # end def

  '''
  Return the required number of inputs
  '''
  def numberOfOutputs( self ):
    if len( self.m_W ) > 0:
      return self.m_W[ -1 ].shape[ 0 ]
    else:
      return 0
    # end if
  # end def

  '''
  Assign parameters
  '''
  def setParameters( self, p ):
    pass
  # end def

  '''
  Add a displacement to parameters
  '''
  def moveParameters( self, d ):
    o = 0
    L = len( self.m_W )
    for l in range( L ):
      b = self.m_W[ l ].shape[ 0 ]
      w = self.m_W[ l ].shape[ 1 ] * b
      self.m_W[ l ] += d[ o : o + w ].reshape( self.m_W[ l ].shape )
      self.m_B[ l ] += d[ o + w : o + w + b ].reshape( self.m_B[ l ].shape )
    # end for
  # end def

  '''
  Return a string-based representation of the model
  '''
  def __str__( self ):
    L = len( self.m_W )
    buf = str( L ) + '\n'
    for i in range( L ):
      buf += str( self.m_W[ i ].shape[ 0 ] ) + ' '
      buf += str( self.m_W[ i ].shape[ 1 ] ) + ' '
      buf += str( self.m_S[ i ] ) + '\n'
    # end for
    p = self.parameters( )
    if not p is None:
      b = io.BytesIO( )
      numpy.savetxt( b, p, fmt = '%.4e', newline = ' ', encoding = 'latin1' )
      buf += b.getvalue( ).decode( 'latin1' )[ 0 : -1 ]
    else:
      buf += 'random'
    # end if
    return buf
  # end def

  '''
  Apply a threshold when possible
  '''
  def threshold( self, X ):
    return None
  # end def

  '''
  Real evaluate method
  '''
  def _evaluate( self, X ):
    assert len( self.m_W ) > 0, 'Parameters should be defined.'
    z = X.T
    for l in range( len( self.m_W ) ):
      z = self.m_S[ l ]( ( self.m_W[ l ] @ z ) + self.m_B[ l ] )
    # end for
    return z.T
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
    def _evaluate( self, samples, need_gradient ):
      X = samples[ 0 ]
      Y = samples[ 1 ]

      # 1. Forward
      A = [ X.T ]
      Z = []
      L = self.m_Model.numberOfLayers( )
      for i in range( L ):
        Z += [ ( self.m_Model.m_W[ i ] @ A[ i ] ) + self.m_Model.m_B[ i ] ]
        A += [ self.m_Model.m_S[ i ]( Z[ i ] ) ]
      # end for

      # 2. Cost
      J = float( 0 )
      last_activation = str( self.m_Model.m_S[ -1 ] )
      if last_activation == 'SoftMax':
        for k in range( Y.shape[ 1 ] ):
          J -= numpy.log( A[ -1 ][ : , Y[ : , k ] == 1 ] + 1e-12 ).sum( )
          J -= numpy.log( 1.0 - A[ -1 ][ : , Y[ : , k ] == 0 ] + 1e-12 ).sum( )
        # end for
        J /= float( X.shape[ 0 ] )
      elif last_activation == 'Sigmoid':
        a = A[ -1 ].T
        J -= numpy.log( a[ Y == 1 ] + 1e-12 ).sum( )
        J -= numpy.log( 1 - a[ Y == 0 ] + 1e-12 ).sum( )
        J /= float( X.shape[ 0 ] )
      else:
        pass
      # end if

      # 3. Backpropagation
      g = None
      if need_gradient:

        # 3.0 Some values
        m = float( X.shape[ 0 ] )

        # 3.1 Compute deltas
        D = []
        if last_activation == 'SoftMax' or last_activation == 'Sigmoid':
          D += [ ( A[ -1 ] - Y.T ) / m ]
        else:
          pass
        # end if
        for l in range( L - 2, -1, -1 ):
          D += [
            numpy.multiply(
              self.m_Model.m_W[ l + 1 ].T @ D[ -1 ],
              self.m_Model.m_S[ l ]( Z[ l ], True )
              )
            ]
        # end for

        # 3.2 Compute derivatives
        t = []
        for l in range( L ):
          t += [ ( D[ L - 1 - l ] @ A[ l ].T ).flatten( ) ]
          t += [ D[ L - 1 - l ].mean( axis = 1 ).flatten( ) ]
        # end for
        g = numpy.concatenate( tuple( t ) ).\
            reshape( ( self.m_Model.numberOfParameters( ), 1 ) )
      # end if

      # Finish
      return [ J, g ]
    # end def

  # end class

# end class

## eof - $RCSfile$
