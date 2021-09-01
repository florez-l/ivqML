## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

'''
'''
class FeedForward:

  '''
  '''
  m_ValidTypes  = ( int, float, list, numpy.matrix, numpy.ndarray )

  '''
  Weight matrices. These matrices are kept in tranposed form.
  '''
  m_Weights     = []

  '''
  '''
  m_Biases      = []

  '''
  '''
  m_Activations = []

  '''
  '''
  m_InputSize   = 0

  '''
  '''
  def __init__( self, input_size ):
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
      input_size = self.m_Weights[ -1 ].shape[ 0 ]
    # end if
    self.m_Activations += [ activation ]

    self.m_Weights += [ numpy.zeros( ( output_size, input_size ) ) ]
    self.m_Biases  += [ numpy.zeros( ( 1, output_size ) ) ]

    print( type( theta ) )

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
      return self.m_Weights[ i ].shape[ 1 ]
    else:
      return 0
    # end if
  # end def

  '''
  '''
  def GetLayerOutputSize( self, i ):
    if i < len( self.m_Weights ):
      return self.m_Weights[ i ].shape[ 0 ]
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
  def __str__( self ):
    r  = '***********************************\n'
    r += '*** Feed forward neural network ***\n'
    for i in range( len( self.m_Weights ) ):
      r += 'Layer ' + str( i ) + '\n'
      r += '  Input size  = ' + str( self.m_Weights[ i ].shape[ 1 ] ) + '\n'
      r += '  Output size = ' + str( self.m_Weights[ i ].shape[ 0 ] ) + '\n'
      r += '  Activation  = ' + str( self.m_Activations[ i ] ) + '\n'
    # end for
    r += '***********************************\n'
    return r
  # end def

# end class

## eof - $RCSfile$
