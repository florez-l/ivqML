## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

'''
'''
class Base:

  '''
  '''
  m_ValidTypes = ( int, float, list, numpy.matrix, numpy.ndarray )

  '''
  '''
  def __init__( self, **kwargs ):
    pass
  # end def

  '''
  '''
  def GetInputSize( self ):
    raise NotImplementedError( )
  # end def

  '''
  '''
  def GetOutputSize( self ):
    raise NotImplementedError( )
  # end def

  '''
  '''
  def SetParameters( self, T ):
    raise NotImplementedError( )
  # end def

  '''
  '''
  def _RealCall( self, x ):
    raise NotImplementedError( )
  # end if

  '''
  '''
  def __call__( self, *cargs, **kwargs ):
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

    return self._RealCall( x, **kwargs )
  # end def

# end class

## eof - $RCSfile$
