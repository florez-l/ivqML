## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy

class Perceptron( object ):
  ## -----------------------------------------------------------------------
  '''Initialize a perceptron with weights, bias and activation function.'''
  def __init__( self, w, b, s = math.tanh, t = 0.0 ):
    assert isinstance( b, ( int, float ) ) , "Invalid bias data type." 
    assert isinstance( t, ( int, float ) ) , "Invalid threshold data type." 

    if type( w ) is int:
      self.m_W = numpy.matrix( [ w ] )
    elif type( w ) is float:
      self.m_W = numpy.matrix( [ w ] )
    elif type( w ) is list:
      self.m_W = numpy.matrix( w )
    elif type( w ) is numpy.matrix:
      self.m_W = w
    else:
      raise TypeError( 'Invalid weights type.' )
    # end if
    assert self.m_W.shape[ 0 ] == 1, "Weights should be a row vector."
    self.m_B = float( b )
    self.m_S = s
    self.m_T = float( t )
  # end def __init__

  ## -----------------------------------------------------------------------
  '''Parentheses operator overloading'''
  def __call__( self, x ):
    if type( x ) is int:
      w = self.m_W @ numpy.matrix( [ float( x ) ] ).T
    elif type( x ) is float:
      w = self.m_W @ numpy.matrix( [ x ] ).T
    elif type( x ) is list:
      w = self.m_W @ numpy.matrix( x ).T
    elif type( w ) is numpy.matrix:
      w = self.m_W @ x
    else:
      raise TypeError( 'Invalid input type.' )
    # end if
    return self.m_S( w.item( ) + self.m_B )
  # end def __call__
# end class

## eof - $RCSfile$
