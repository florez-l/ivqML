## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys

'''
'''
class MaximumLikelihood:

  '''
  '''
  m_Eps = 1e-8

  '''
  '''
  def __init__( self, in_X, in_y ):
    assert isinstance( in_X, ( list, numpy.matrix, numpy.ndarray ) ), \
      "Invalid X type."
    assert isinstance( in_y, ( list, numpy.matrix, numpy.ndarray ) ), \
      "Invalid y type."
    
    if type( in_X ) is list:
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

    self.m_Xby = \
      ( numpy.array( self.m_X ) * numpy.array( self.m_y ) ).mean( axis = 0 )
    self.m_uy = self.m_y.mean( )
  # end def

  def NumberOfExamples( self ):
    return self.m_M
  # end def

  def VectorSize( self ):
    return self.m_N + 1
  # end def

  '''
  '''
  def CostAndGradient( self, theta ):
    b = theta[ : , 0 ]
    w = theta[ : , 1 : ]
    z = 1.0 / ( 1.0 + numpy.exp( -( ( self.m_X @ w.T ) + b ) ) )
    p = numpy.log(
      z[ numpy.where( self.m_y[ : , 0 ] == 1 )[ 0 ] , : ] + self.m_Eps
      ).sum( )
    n = numpy.log(
      1 - z[ numpy.where( self.m_y[ : , 0 ] == 0 )[ 0 ] , : ] + self.m_Eps
      ).sum( )
    J = -( p + n ) / self.m_M
    dw = numpy.matrix(
      ( numpy.array( self.m_X ) * numpy.array( z ) ).mean( axis = 0 ) -
        self.m_Xby
      )
    db = numpy.matrix( z.mean( axis = 0 ) - self.m_uy )

    return [ J, numpy.concatenate( ( db, dw ), axis = 1 ) ]
  # end def

# end class

## eof - $RCSfile$
