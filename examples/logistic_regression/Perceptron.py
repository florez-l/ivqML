## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy

## -------------------------------------------------------------------------
class Perceptron:

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  '''Initialize a perceptron with weights, bias and activation function.'''
  def __init__( self, w, b, s ):
    assert isinstance( b, ( int, float ) ) , "Invalid bias data type."

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
    self.m_T = self.m_S.Threshold
  # end def __init__

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  '''Parentheses operator overloading'''
  def __call__( self, x ):
    if type( x ) is int:
      w = self.m_W @ numpy.matrix( [ float( x ) ] ).T
    elif type( x ) is float:
      w = self.m_W @ numpy.matrix( [ x ] ).T
    elif type( x ) is list:
      w = self.m_W @ numpy.matrix( x ).T
    elif type( x ) is numpy.matrix:
      w = self.m_W @ x.T
    else:
      raise TypeError( 'Invalid input type.' )
    # end if
    return self.m_S( w.item( ) + self.m_B )
  # end def __call__

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  '''.'''
  def __getitem__( self, x ):
    if self( x ) < self.m_T:
      return False
    else:
      return True
  # end def

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  '''Create a resulting image'''
  def create_example_image( self, filename, X, Y, sampling_w = 100 ):
    min_w = X.min( axis = 0 )
    max_w = X.max( axis = 0 )

    # Fill labels
    dif_w = max_w - min_w
    off_w = dif_w / ( sampling_w - 1 )
    image = [ 0 for i in range( sampling_w * sampling_w * 3 ) ]
    k = 0
    for s2 in range( sampling_w ):
      w2 = ( off_w[ 0, 1 ] * float( s2 ) ) + min_w[ 0, 1 ]
      for s1 in range( sampling_w ):
        w1 = ( off_w[ 0, 0 ] * float( s1 ) ) + min_w[ 0, 0 ]
        if self[ [ w1, w2 ] ]:
          image[ k ] = 255
          image[ k + 1 ] = 0
          image[ k + 2 ] = 0
        else:
          image[ k ] = 0
          image[ k + 1 ] = 0
          image[ k + 2 ] = 255
        # end if
        k = k + 3
      # end for
    # end for

    # Fill samples
    for i in range( X.shape[ 0 ] ):
      try:
        x = float( sampling_w - 1 ) * ( ( X[ i ] - min_w ) / dif_w )
        j = 3 * ( int( x[ 0, 0 ] ) + int( x[ 0, 1 ] ) * sampling_w )
        image[ j + 0 ] = 255 * int( Y[ i, 0 ] )
        image[ j + 1 ] = 255 * int( Y[ i, 0 ] )
        image[ j + 2 ] = 255 * int( Y[ i, 0 ] )
      except IndexError:
        pass
      # end try
    # end for
    
    # Save image
    with open( filename, "wb" ) as out_image:
      out_image.write( b"P6\n" )
      out_image.write( b"# Created as a logistic regression result\n" )
      out_image.write(
        bytearray(
          "{:d} {:d}\n".format( sampling_w, sampling_w ),
          encoding="ascii"
          )
        )
      out_image.write( b"255\n" )
      out_image.write( bytearray( image ) )
    # end with
  # end def

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  '''.'''
  def confussion_matrix( self, X, Y ):
    M = numpy.zeros( ( 2, 2 ) )
    for i in range( X.shape[ 0 ] ):
      M[ self[ X[ i ] ] * 1, int( Y[ i, 0 ] ) ] += 1
    # end for
    TP = M[ 0, 0 ]
    TN = M[ 1, 1 ]
    FP = M[ 0, 1 ]
    FN = M[ 1, 0 ]
    print( "--------------------" )
    print( "Confussion matrix:\n", M )
    print( "Sensitivity : {:.2f}%".format( 100.0 * TP / ( TP + FN ) ) )
    print( "Specificity : {:.2f}%".format( 100.0 * TN / ( TN + FP ) ) )
    print( "Precision   : {:.2f}%".format( 100.0 * TP / ( TP + FP ) ) )
    print( "Accuracy    : {:.2f}%".format( 100.0 * ( TP + TN ) / ( TP + TN + FP + FN ) ) )
    print( "F1 score    : {:.2f}%".format( 100.0 * ( 2 * TP ) / ( ( 2 * TP ) + FP + FN ) ) )
    print( "--------------------" )
  # end def

# end class

## eof - $RCSfile$
