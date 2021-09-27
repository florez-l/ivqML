## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, re
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
  m_Weights     = []
  m_Biases      = []
  m_Activations = []
  m_InputSize   = 0

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
    if len( self.m_Weights ) > 0:
      input_size = self.m_Weights[ -1 ].shape[ 1 ]
    # end if

    if isinstance( activation, ( str ) ):
      self.m_Activations += [ getattr( Activation, activation )( ) ]
    else:
      self.m_Activations += [ activation ]
    # end if

    if theta is None:
      self.m_Weights += [ numpy.zeros( ( input_size, output_size ) ) ]
      self.m_Biases  += [ numpy.zeros( ( 1, output_size ) ) ]
    elif isinstance( theta, ( tuple ) ):
      if len( theta ) == 2:
        w, b = theta
        assert \
               w.shape == ( input_size, output_size ) and \
               b.shape == ( 1, output_size ), \
               'Invalid sizes'
        self.m_Weights += [ w ]
        self.m_Biases  += [ b ]
      else:
        pass
      # end if
    elif isinstance( theta, ( str ) ):
      if theta == 'random':
        self.m_Weights += \
          [ numpy.random.uniform( size = ( input_size, output_size ) ) ]
        self.m_Biases += \
          [ numpy.random.uniform( size = ( 1, output_size ) ) ]
      elif theta == 'ones':
        self.m_Weights += [ numpy.ones( ( input_size, output_size ) ) ]
        self.m_Biases  += [ numpy.ones( ( 1, output_size ) ) ]
      elif theta == 'zeros':
        self.m_Weights += [ numpy.zeros( ( input_size, output_size ) ) ]
        self.m_Biases  += [ numpy.zeros( ( 1, output_size ) ) ]
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
    return len( self.m_Weights )
  # end def

  '''
  '''
  def GetLayerInputSize( self, i ):
    if i < len( self.m_Weights ):
      return self.m_Weights[ i ].shape[ 0 ]
    else:
      return 0
    # end if
  # end def

  '''
  '''
  def GetLayerOutputSize( self, i ):
    if i < len( self.m_Weights ):
      return self.m_Weights[ i ].shape[ 1 ]
    else:
      return 0
    # end if
  # end def

  '''
  '''
  def GetLayerWeights( self, i ):
    if i < len( self.m_Weights ):
      return self.m_Weights[ i ]
    else:
      return None
    # end if
  # end def

  '''
  '''
  def GetLayerBiases( self, i ):
    if i < len( self.m_Biases ):
      return self.m_Biases[ i ]
    else:
      return None
    # end if
  # end def

  '''
  '''
  def GetLayerActivation( self, i ):
    if i < len( self.m_Activations ):
      return self.m_Activations[ i ]
    else:
      return None
    # end if
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

    for i in range( len( self.m_Weights ) ):
      hnd.write( str( self.m_Weights[ i ].shape[ 0 ] ) + ' ' )
      hnd.write( str( self.m_Weights[ i ].shape[ 1 ] ) + '\n' )
      numpy.savetxt( hnd, self.m_Weights[ i ], fmt = '%.8e' )
      numpy.savetxt( hnd, self.m_Biases[ i ], fmt = '%.8e' )
      hnd.write( str( self.m_Activations[ i ] ) + '\n' )
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
    assert x.shape[ 1 ] == self.m_InputSize, 'Input size is not compatible.'

    z = ( x @ self.m_Weights[ 0 ] ) + self.m_Biases[ 0 ]
    a = self.m_Activations[ 0 ]( z )
    for i in range( 1, len( self.m_Weights ) ):
      z = ( a @ self.m_Weights[ i ] ) + self.m_Biases[ i ]
      a = self.m_Activations[ i ]( z )
    # end for
    return a
  # end def

  '''
  '''
  def BackPropagate( self, X, Y, propagation_type = 'cce' ):

    # Forward propagation
    A = [ X ]
    Z = []
    for i in range( len( self.m_Weights ) ):
      Z += [ ( A[ i ] @ self.m_Weights[ i ] ) + self.m_Biases[ i ] ]
      A += [ self.m_Activations[ i ]( Z[ i ] ) ]
    # end for

    # Compute last layer delta
    D = None
    d = numpy.array( A[ -1 ] - Y )
    if propagation_type == 'mse':
      dL = \
         numpy.array( d ) * \
         numpy.array( self.m_Activations[ i ]( Z[ -1 ] ), derivative = True )
      D = [ 2.0 * dL ]
    elif propagation_type == 'bce' or propagation_type == 'cce':
      D = [ d ]
    else:
      raise TypeError( 'Invalid propagation type (' + propagation_type + ')' )
    # end if

    # Compute other layers delta
    L = self.GetNumberOfLayers( )
    for l in range( L - 2, -1, -1 ):
      d = \
        numpy.array( self.m_Weights[ l + 1 ] @ D[ L - l - 2 ].T ).T * \
        numpy.array( self.m_Activations[ l ]( Z[ l ], derivative = True ) )
      D += [ d ]
    # end for
    D.reverse( )

    # Flatten matrices
    G = None
    for l in range( L ):
      gW = ( ( A[ l ].T @ D[ l ] ) / float( X.shape[ 0 ] ) ).flatten( )
      if G is None:
        G = gW
      else:
        G = numpy.append( G, gW )
      # end if
      G = numpy.append( G, D[ l ].mean( axis = 0 ).flatten( ) )
    # end for

    return G

  # end def

  '''
  '''
  def __str__( self ):
    r  = '***********************************\n'
    r += '*** Feed forward neural network ***\n'
    for i in range( len( self.m_Weights ) ):
      r += 'Layer ' + str( i ) + '\n'
      r += '  Input size  = ' + str( self.m_Weights[ i ].shape[ 0 ] ) + '\n'
      r += '  Output size = ' + str( self.m_Weights[ i ].shape[ 1 ] ) + '\n'
      r += '  Activation  = ' + str( self.m_Activations[ i ] ) + '\n'
    # end for
    r += '***********************************'
    return r
  # end def

# end class

## eof - $RCSfile$
