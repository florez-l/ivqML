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
  def __init__( self, w, b = None, o = None, D = None, T = None ):

    # -- Get model weights and bias
    assert isinstance( w, self.m_ValidTypes ), \
           'Invalid type for model weights: ' + str( type( w ) )
    if b is None:
      theta = numpy.matrix( w )
      if theta.shape[ 1 ] > 1:
        theta = theta.T
      # end if
      mb = theta[ 0, 0 ]
      mw = theta[ 1 : , : ]
    else:
      assert isinstance( b, ( int, float ) )
      mb = float( b )
      mw = numpy.matrix( w )
      if mw.shape[ 1 ] > 1:
        mw = mw.T
      # end if
    # end if
    assert mw.shape[ 1 ] == 1, 'Invalid weights size: ' + str( mw.shape )

    # -- Process projection matrix
    if not T is None:
      assert isinstance( T, self.m_ValidTypes ), \
             'Invalid type for model projection: ' + str( type( T ) )
      mT = numpy.matrix( T )
      n = mT.shape[ 0 ]
      p = mT.shape[ 1 ]
      if p > n:
        mT = mT.transpose
        n, p = p, n
      # end if
    else:
      n = p = mw.shape[ 0 ]
      mT = numpy.identity( n )
    # end if
    assert mT.shape[ 1 ] == mw.shape[ 0 ], \
           'Invalid projection vs. weights size: ' + \
           str( mT.shape ) +  ' ' + \
           str( mw.shape )

    # -- Process scale matrix
    if not D is None:
      assert isinstance( D, self.m_ValidTypes ), \
        'Invalid type for model scale: ' + str( type( D ) )
      mD = numpy.diag( D )
    else:
      mD = numpy.identity( p )
    # end if
    assert mD.shape[ 0 ] == p and mD.shape[ 1 ] == p, \
           'Invalid scale size: ' + str( mD.shape ) +  ' != ' + str( p )

    # -- Process offset vector
    if not o is None:
      assert isinstance( w, self.m_ValidTypes ), \
             'Invalid type for model weights: ' + str( type( w ) )
      mo = numpy.matrix( o )
      if mo.shape[ 0 ] > 1:
        mo = mo.T
      # end if
    else:
      mo = numpy.zeros( ( 1, p ) )
    # end if
    assert mo.shape[ 0 ] == 1 and mo.shape[ 1 ] == p, \
           'Invalid offset size: ' + str( no.shape )

    # -- Configure real values
    Dw = numpy.linalg.inv( mD ) @ mw
    self.m_Weights =  mT @ Dw
    self.m_Bias = mb - ( mo @ Dw )
  # end def

  '''
  '''
  def Dimensions( self ):
    return self.m_Weights.shape[ 0 ]
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

    assert x.shape[ 1 ] == self.m_Weights.shape[ 0 ], \
      'Input size is not compatible.'

    return ( x @ self.m_Weights ) + self.m_Bias
  # end def

# end class

## eof - $RCSfile$
