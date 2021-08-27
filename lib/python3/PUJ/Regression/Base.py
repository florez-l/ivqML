## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

'''
'''
class Base:

  '''
  '''
  def __init__( self, in_X, in_y, in_Xtra = None, in_ytra = None ):
    assert isinstance( in_X, ( list, numpy.matrix, numpy.ndarray ) ), \
      "Invalid X type."
    assert isinstance( in_y, ( list, numpy.matrix, numpy.ndarray ) ), \
      "Invalid y type."
    
    if type( in_X ) is list or type( in_X ) is numpy.ndarray:
      self.m_X = numpy.matrix( in_X )
    else:
      self.m_X = in_X
    # end if
    if type( in_y ) is list:
      self.m_y = numpy.matrix( in_y ).T
    else:
      self.m_y = in_y
    # end if
    assert self.m_X.shape[ 0 ] == self.m_y.shape[ 0 ], "Invalid X,y sizes."
    assert self.m_y.shape[ 1 ] == 1, "Invalid y size."

    self.m_M = self.m_X.shape[ 0 ]
    self.m_N = self.m_X.shape[ 1 ]
  # end def

  '''
  '''
  def NumberOfExamples( self ):
    return self.m_M
  # end def

  '''
  '''
  def VectorSize( self ):
    return self.m_N + 1
  # end def

# end class

## eof - $RCSfile$
