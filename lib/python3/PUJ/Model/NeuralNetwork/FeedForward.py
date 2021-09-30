## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy, re
import PUJ.Model.NeuralNetwork.Activation as Activation

'''
'''
class FeedForward:

  '''
  '''
  m_ValidTypes  = ( int, float, list, numpy.matrix, numpy.ndarray )

  '''
  Weight matrices are kept in tranposed form.
  '''
  m_W = []
  m_B = []
  m_S = []
  m_InputSize = 0

  '''
  '''
  def __init__( self, input_size = 1 ):
    assert isinstance( input_size, ( int ) ) and input_size > 0, \
           'Invalid input size'
    self.m_InputSize = input_size
  # end def

  '''
  '''
  def AddLayer( self, output_size, activation, theta = None ):
    assert isinstance( output_size, ( int ) ) and output_size > 0, \
           'Invalid output size'

    input_size = self.m_InputSize
    if len( self.m_W ) > 0:
      input_size = self.m_W[ -1 ].shape[ 1 ]
    # end if

    if isinstance( activation, ( str ) ):
      self.m_S += [ getattr( Activation, activation )( ) ]
    else:
      self.m_S += [ activation ]
    # end if

    if theta is None:
      self.m_W += [ numpy.zeros( ( input_size, output_size ) ) ]
      self.m_B  += [ numpy.zeros( ( 1, output_size ) ) ]
    elif isinstance( theta, ( tuple ) ):
      if len( theta ) == 2:
        w, b = theta
        assert \
               w.shape == ( input_size, output_size ) and \
               b.shape == ( 1, output_size ), \
               'Invalid sizes'
        self.m_W += [ w ]
        self.m_B  += [ b ]
      else:
        pass
      # end if
    elif isinstance( theta, ( str ) ):
      if theta == 'random':
        self.m_W += \
          [ numpy.random.uniform( size = ( input_size, output_size ) ) ]
        self.m_B += \
          [ numpy.random.uniform( size = ( 1, output_size ) ) ]
      elif theta == 'ones':
        self.m_W += [ numpy.ones( ( input_size, output_size ) ) ]
        self.m_B  += [ numpy.ones( ( 1, output_size ) ) ]
      elif theta == 'zeros':
        self.m_W += [ numpy.zeros( ( input_size, output_size ) ) ]
        self.m_B  += [ numpy.zeros( ( 1, output_size ) ) ]
      else:
        pass
      # end if
    else:
      pass
    # end if

  # end def

  '''
  '''
  def GetNumberOfLayers( self ):
    return len( self.m_W )
  # end def

  '''
  '''
  def GetLayerInputSize( self, i ):
    if i < len( self.m_W ):
      return self.m_W[ i ].shape[ 0 ]
    else:
      return 0
    # end if
  # end def

  '''
  '''
  def GetLayerOutputSize( self, i ):
    if i < len( self.m_W ):
      return self.m_W[ i ].shape[ 1 ]
    else:
      return 0
    # end if
  # end def

  '''
  '''
  def GetLayerWeights( self, i ):
    if i < len( self.m_W ):
      return self.m_W[ i ]
    else:
      return None
    # end if
  # end def

  '''
  '''
  def GetLayerBiases( self, i ):
    if i < len( self.m_B ):
      return self.m_B[ i ]
    else:
      return None
    # end if
  # end def

  '''
  '''
  def GetLayerActivation( self, i ):
    if i < len( self.m_S ):
      return self.m_S[ i ]
    else:
      return None
    # end if
  # end def

  '''
  '''
  def GetNumberOfParameters( self ):
    n = 0
    for w in self.m_W:
      n += ( w.shape[ 0 ] + 1 ) * w.shape[ 1 ]
    # end for
    return n
  # end def

  '''
  '''
  def SetParameters( self, theta ):
    i = 0
    for l in range( len( self.m_W ) ):

      ws = self.m_W[ l ].shape
      wl = i + ws[ 0 ] * ws[ 1 ]
      bl = ws[ 1 ]
      bs = ( 1, bl )

      self.m_W[ l ] = theta[ i : wl ].reshape( ws )
      self.m_B[ l ] = theta[ wl : wl + bl ].reshape( bs )

      i = wl + bl
      
    # end for
  # end def

  '''
  '''
  def Flatten( self ):
    f = None
    for l in range( len( self.m_W ) ):
      if f is None:
        f = self.m_W[ l ].flatten( )
      else:
        f = numpy.append( f, self.m_W[ l ].flatten( ) )
      # end if
      f = numpy.append( f, self.m_B[ l ].flatten( ) )
    # end for
    return f
  # end def

  '''
  '''
  def LoadParameters( self, fname_or_handle ):

    if isinstance( fname_or_handle, ( str ) ):
      hnd = open( fname_or_handle, 'r' )
    else:
      hnd = fname_or_handle
    # end if

    data = re.sub(
      ' +', ' ', ''.join( hnd.readlines( ) ).replace( '\n', ' ' )
      ).split( )

    is_draft = False
    i = 0
    if data[ 0 ] == 'draft':
      is_draft = True
      i = 1
    # end if

    in_size = int( data[ i ] )
    self.m_InputSize = in_size
    out_size = int( data[ i + 1 ] )
    i += 2
    while in_size > 0 and out_size > 0:

      if not is_draft:

        is_os = in_size * out_size
        wl = [ float( v ) for v in data[ i : i + is_os ] ]
        bl = [ float( v ) for v in data[ i + is_os : i + is_os + out_size ] ]
        w = numpy.matrix( wl ).reshape( ( in_size, out_size ) )
        b = numpy.matrix( bl ).reshape( ( 1, out_size ) )
        a = data[ i + is_os + out_size ]

        self.AddLayer( out_size, a, ( w, b ) )
        i += is_os + out_size + 3

      else:

        a = data[ i ]
        self.AddLayer( out_size, a, 'random' )
        i += 1

      # end if

      in_size = int( data[ i ] )
      out_size = int( data[ i + 1 ] )
      i += 2

    # end while

    if isinstance( fname_or_handle, ( str ) ):
      hnd.close( )
    # end if

  # end def

  '''
  '''
  def SaveParameters( self, fname_or_handle ):

    if isinstance( fname_or_handle, ( str ) ):
      hnd = open( fname_or_handle, 'w' )
    else:
      hnd = fname_or_handle
    # end if

    for i in range( len( self.m_W ) ):
      hnd.write( str( self.m_W[ i ].shape[ 0 ] ) + ' ' )
      hnd.write( str( self.m_W[ i ].shape[ 1 ] ) + '\n' )
      numpy.savetxt( hnd, self.m_W[ i ], fmt = '%.8e' )
      numpy.savetxt( hnd, self.m_B[ i ], fmt = '%.8e' )
      hnd.write( str( self.m_S[ i ] ) + '\n' )
    # end for
    hnd.write( '0 0\n' )

    if isinstance( fname_or_handle, ( str ) ):
      hnd.close( )
    # end if

  # end def

  '''
  '''
  def __call__( self, *cargs ):
    assert len( cargs ) > 0, 'No arguments passed to __call__()'
    if len( cargs ) == 1:
      assert isinstance( cargs[ 0 ], self.m_ValidTypes ), 'Invalid types'
      if isinstance( cargs[ 0 ], ( numpy.matrix ) ):
        x = cargs[ 0 ]
      else:
        x = numpy.matrix( cargs[ 0 ] )
      # end if
    else:
      x = numpy.matrix( list( cargs ) )
    # end if
    assert x.shape[ 1 ] == self.m_W[ 0 ].shape[ 0 ], \
           'Input size is not compatible.'

    z = ( x @ self.m_W[ 0 ] ) + self.m_B[ 0 ]
    a = self.m_S[ 0 ]( z )
    for i in range( 1, len( self.m_W ) ):
      z = ( a @ self.m_W[ i ] ) + self.m_B[ i ]
      a = self.m_S[ i ]( z )
    # end for
    return a
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
      assert model.GetLayerInputSize( 0 ) == in_X.shape[ 1 ], \
             'Invalid input size'
      
      self.m_X = in_X
      self.m_Y = in_Y
      self.m_Model = model
      self.SetPropagationTypeToMinimumSquareError( )
    # end def

    '''
    '''
    def SetPropagationTypeToMinimumSquareError( self ):
      self.m_Propagation = 'mse'
    # end def

    '''
    '''
    def SetPropagationTypeToBinaryCrossEntropy( self ):
      self.m_Propagation = 'bce'
    # end def

    '''
    '''
    def SetPropagationTypeToCategoricalCrossEntropy( self ):
      self.m_Propagation = 'cce'
    # end def


    def VectorSize( self ):
      return self.m_Model.GetNumberOfParameters( )
    # end def

    '''
    '''
    def GetInitialParameters( self ):
      return self.m_Model.Flatten( )
    # end def

    '''
    '''
    def GetModel( self ):
      return self.m_Model
    # end def

    '''
    '''
    def _Cost( self, Yp ):
      J = math.inf
      if self.m_Propagation == 'mse':
        d = Yp - self.m_Y
        J = ( d.T @ d )[ 0, 0 ] / float( self.m_Y.shape[ 0 ] )
      elif self.m_Propagation == 'bce':
        p = numpy.log(
          Yp[ numpy.where( self.m_Y[ : , 0 ] == 1 )[ 0 ] , : ] + self.m_Eps
          ).sum( )
        n = numpy.log(
          1 - Yp[ numpy.where( self.m_Y[ : , 0 ] == 0 )[ 0 ] , : ] + self.m_Eps
          ).sum( )
        J = -( p + n ) / float( self.m_Y.shape[ 0 ] )
      elif self.m_Propagation == 'cce':
        pass
      # end if

      return J
    # end def

    '''
    '''
    def Cost( self, theta ):
      self.m_Model.SetParameters( theta )
      return self._Cost( self.m_Model( self.m_X ) )
    # end def

    '''
    '''
    def CostAndGradient( self, theta ):
      self.m_Model.SetParameters( theta )

      # Forward propagation
      A = [ self.m_X ]
      Z = []
      for l in range( len( self.m_Model.m_W ) ):
        Z += [ ( A[ l ] @ self.m_Model.m_W[ l ] ) + self.m_Model.m_B[ l ] ]
        A += [ self.m_Model.m_S[ l ]( Z[ l ] ) ]
      # end for
      J = self._Cost( A[ -1 ] )

      # Compute last layer delta
      D = None
      d = numpy.array( A[ -1 ] - self.m_Y )
      if self.m_Propagation == 'mse':
        d = \
          2.0 * \
          numpy.array( d ) * \
          numpy.array( self.m_Model.m_S[ -1 ]( Z[ -1 ] ), derivative = True )
        D = [ d ]
      elif self.m_Propagation == 'bce' or self.m_Propagation == 'cce':
        D = [ d ]
      else:
        raise TypeError( 'Invalid propagation type (' + self.m_Propagation + ')' )
      # end if

      # Compute other layers deltas
      L = len( self.m_Model.m_W )
      for l in range( L - 2, -1, -1 ):
        d = \
          numpy.array( self.m_Model.m_W[ l + 1 ] @ D[ L - l - 2 ].T ).T * \
          numpy.array( self.m_Model.m_S[ l ]( Z[ l ], derivative = True ) )
        D += [ d ]
      # end for
      D.reverse( )

      # Flatten matrices
      G = None
      for l in range( L ):
        gW = ( ( A[ l ].T @ D[ l ] ) / float( self.m_X.shape[ 0 ] ) ).flatten( )
        if G is None:
          G = gW
        else:
          G = numpy.append( G, gW )
        # end if
        G = numpy.append( G, D[ l ].mean( axis = 0 ).flatten( ) )
      # end for

      return [ J, G ]

    # end def
  # end class

  '''
  '''
  def __str__( self ):
    r  = '***********************************\n'
    r += '*** Feed forward neural network ***\n'
    for i in range( len( self.m_W ) ):
      r += 'Layer ' + str( i ) + '\n'
      r += '  Input size  = ' + str( self.m_W[ i ].shape[ 0 ] ) + '\n'
      r += '  Output size = ' + str( self.m_W[ i ].shape[ 1 ] ) + '\n'
      r += '  Activation  = ' + str( self.m_S[ i ] ) + '\n'
    # end for
    r += '***********************************'
    return r
  # end def

# end class

## eof - $RCSfile$
