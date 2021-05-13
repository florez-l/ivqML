## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

## -------------------------------------------------------------------------
'''
'''
class MSE:

  '''
  '''
  def __init__( self, in_X, in_y ):
    assert isinstance( in_X, ( list, numpy.matrix ) ), "Invalid X type."
    assert isinstance( in_y, ( list, numpy.matrix ) ), "Invalid Y type."
    
    if type( in_X ) is list:
      X = numpy.matrix( in_X )
    else:
      X = in_X
    # end if
    if type( in_y ) is list:
      y = numpy.matrix( in_y ).T
    else:
      y = in_y
    # end if
    assert X.shape[ 0 ] == y.shape[ 0 ], "Invalid X,Y sizes."
    assert y.shape[ 1 ] == 1, "Invalid Y size."

    self.m_M = X.shape[ 0 ]
    self.m_N = X.shape[ 1 ]

    self.m_XtX = ( X.T / float( self.m_M ) ) @ X
    self.m_Xby = ( numpy.array( X ) * numpy.array( y ) ).mean( axis = 0 )
    self.m_uX = X.mean( axis = 0 )
    self.m_uy = y.mean( )
    self.m_yty = ( y.T / float( self.m_M ) ) @ y
  # end def

  def NumberOfExamples( self ):
    return self.m_M
  # end def

  def VectorSize( self ):
    return self.m_N
  # end def

  '''
  '''
  def AnalyticSolve( self ):
      x = numpy.append( self.m_Xby, numpy.array( [ self.m_uy ] ), axis = 0 )
      B = numpy.append( self.m_uX, numpy.matrix( [ 1 ] ), axis = 1 )
      A = numpy.append( self.m_XtX, self.m_uX.T, axis = 1 )
      A = numpy.append( A, B, axis = 0 )
      Wb = x @ numpy.linalg.inv( A )
      return [ Wb[ :, 0 : Wb.shape[ 1 ] - 1 ], Wb[ : , -1 ] ]
  # end def

  '''
  '''
  def CostAndDerivatives( self, W, b ):
    J = \
      ( W @ self.m_XtX @ W.T ) + \
      ( 2.0 * b * ( W @ self.m_uX.T ) ) + \
      ( b * b ) - \
      ( 2.0 * ( W @ self.m_Xby.T ) ) - \
      ( 2.0 * b * self.m_uy ) + \
      self.m_yty
    dW = 2.0 * ( ( W @ self.m_XtX ) + ( b * self.m_uX ) - self.m_Xby )
    db = 2.0 * ( ( W @ self.m_uX.T ) + b - self.m_uy )
    return [ J, dW, db ]
      
  # end def
# end class

## eof - $RCSfile$
