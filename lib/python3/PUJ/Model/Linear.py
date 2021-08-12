## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

'''
'''
class Linear:

  '''
  '''
  m_ValidTypes = ( int, float, list, numpy.matrix, numpy.ndarray )
  m_Weights    = None
  m_Bias       = None

  '''
  '''
  def __init__( self, w, b = None ):
    
    assert isinstance( w, self.m_ValidTypes ), \
      'Invalid type for model weights'

    theta = numpy.matrix( w )
    assert theta.shape[ 0 ] == 1, 'Parameters should be a row vector.'

    mb = None
    if b != None:
      assert isinstance( b, self.m_ValidTypes ), \
        'Invalid type for model bias'

      self.m_Weights = theta
      mb = numpy.matrix( b )

      assert mb.shape[ 0 ] == 1 and mb.shape[ 1 ] == 1, \
        'Invalid parameters sizes.'
      
    else:
      self.m_Weights = theta[ :, 1 : ]
      mb = theta[ :, : -1 ]
    # end if

    self.m_Bias = mb[ 0, 0 ]

  # end def

  '''
  '''
  def Dimensions( self ):
    return self.m_Weights.shape[ 1 ]
  # end def

  '''
  '''
  def __call__( self, *cargs ):
    assert len( cargs ) > 0, 'No arguments passed to __call__()'

    if len( cargs ) == 1:
      assert isinstance( cargs[ 0 ], self.m_ValidTypes )
      if isinstance( cargs[ 0 ], ( numpy.matrix ) ):
        x = cargs[ 0 ]
      else:
        x = numpy.matrix( cargs[ 0 ] )
      # end if
    else:
      x = numpy.matrix( list( cargs ) )
    # end if

    assert x.shape[ 1 ] == self.m_Weights.shape[ 1 ], \
      'Input size is not compatible.'

    return ( ( self.m_Weights @ x.T ) + self.m_Bias ).T

  # end def

# end class

## eof - $RCSfile$
