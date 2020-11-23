## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import csv, math, numpy, random, sys

## -------------------------------------------------------------------------
class BinaryLabelingCost:

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  '''
  Constructor method from file.
  '''
  @classmethod
  def ReadFromFile( cls, filename, s, e, delimiter = "," ):
    fptr = open( filename )
    reader = csv.reader( fptr, delimiter = delimiter )
    line_count = 0
    X = []
    Y = []
    for row in reader:
      try:
        x = [ float( row[ i ] ) for i in range( len( row ) - 1 ) ]
        y = float( row[ len( row ) - 1 ] )
        X.append( x )
        Y.append( y )
      except Exception:
        pass
      # end try
    # end for
    fptr.close( )
    return cls( X, Y, s, e )
  # end def ReadFromFile( )

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  '''
  Constructor method
  @input X input examples as a numpy matrix of m x n dimensions
  @input Y input results as a numpy matrix of m x 1 dimensions
  @output An object created with some intermediary values useful for
          gradient descent.
  '''
  def __init__( self, X, Y, s, e = 1e-10 ):
    assert isinstance( X, ( list, numpy.matrix ) ), "Invalid X type."
    assert isinstance( Y, ( list, numpy.matrix ) ), "Invalid Y type."
    
    if type( X ) is list:
      self.m_X = numpy.matrix( X )
    else:
      self.m_X = X
    # end if
    if type( Y ) is list:
      self.m_Y = numpy.matrix( Y ).T
    else:
      self.m_Y = Y
    # end if
    assert self.m_X.shape[ 0 ] == self.m_Y.shape[ 0 ], "Invalid X,Y sizes."
    assert self.m_Y.shape[ 1 ] == 1, "Invalid Y size."

    self.m_M = self.m_X.shape[ 0 ]
    self.m_N = self.m_X.shape[ 1 ]
    self.m_S = s
    self.m_Epsilon = e
  # end def __init__

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  '''
  Evaluation method
  @input w parameter 1 x n vector
  @input b bias real number
  @output J(w,b)
  '''
  def __call__( self, w, b ): ## trainer( w, b ) <- Cost
    assert isinstance( b, ( int, float, numpy.float128 ) ), \
           "Invalid bias type."

    if type( w ) is int:
      L  = self.m_S( ( numpy.matrix( [ float( w ) ] ) @ self.m_X.T ) + b )
    elif type( w ) is float:
      L  = self.m_S( ( numpy.matrix( [ w ] ) @ self.m_X.T ) + b )
    elif type( w ) is list:
      L  = self.m_S( ( numpy.matrix( w ) @ self.m_X.T ) + b )
    elif type( w ) is numpy.matrix:
      L  = self.m_S( ( w @ self.m_X.T ) + b )
    else:
      raise TypeError( 'Invalid weights type.' )
    # end if

    J = numpy.log(       L[ ( self.m_Y != 0 ).T ] + self.m_Epsilon ).sum( ) + \
        numpy.log( 1.0 - L[ ( self.m_Y == 0 ).T ] + self.m_Epsilon ).sum( )
    if math.isnan( J ):
      raise ValueError( "J is NaN." )
    # end if
    return -J / float( self.m_M )
  # end def __call__

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  '''
  Gradient calculation method
  @input w parameter 1 x n vector
  @input b bias real number
  @output [ dJ(w,b)/dw, dJ(w,b)/db ]
  '''
  def gradient( self, w, b ):
    assert isinstance( b, ( int, float, numpy.float128 ) ), \
           "Invalid bias type."

    if type( w ) is int:
      s = self.m_S( ( numpy.matrix( [ float( w ) ] ) @ self.m_X.T ) + b )
    elif type( w ) is float:
      s = self.m_S( ( numpy.matrix( [ w ] ) @ self.m_X.T ) + b )
    elif type( w ) is list:
      s = self.m_S( ( numpy.matrix( w ) @ self.m_X.T ) + b )
    elif type( w ) is numpy.matrix:
      s = self.m_S( ( w @ self.m_X.T ) + b )
    else:
      raise TypeError( 'Invalid weights type.' )
    # end if

    c = self.m_Y - s.T
    return [ -numpy.multiply( self.m_X, c ).mean( axis = 0 ), -c.mean( ) ]
  # end def gradient

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  '''.'''
  def gradient_descent( self, w0, b0, lr, r, max_iter = -1 ):
    assert isinstance( b0, ( int, float, numpy.float128 ) ), \
           "Invalid bias type."

    if type( w0 ) is int:
      w = numpy.matrix( [ float( w0 ) ] )
    elif type( w0 ) is float:
      w = numpy.matrix( [ w0 ] )
    elif type( w0 ) is list:
      w = numpy.matrix( w0 )
    elif type( w0 ) is numpy.matrix:
      w = w0
    else:
      raise TypeError( 'Invalid weights type.' )
    # end if
    b = b0
    rm = r / float( self.m_M )
    J = self( w, b ) + ( ( ( w @ w.T ).item( ) + ( b * b ) ) * rm )
    Jn = 2.0 * J

    n_iter = 0
    stop = False
    while not stop:
      [ dw, db ] = self.gradient( w, b )
      w = w - ( ( dw + ( w * 2.0 * rm ) ) * lr )
      b = b - ( ( db + ( b * 2.0 * rm ) ) * lr )
      Jn = self( w, b ) + ( ( ( w @ w.T ).item( ) + ( b * b ) ) * rm )

      if n_iter % 10000 == 0:
        print( "Iteration: {: 7d}, dJ = {:.4e}".format( n_iter, J - Jn ) )
      # end if
      if J - Jn < self.m_Epsilon:
        stop = True
      else:
        J = Jn
      # end if
      n_iter += 1
      if n_iter == max_iter:
        stop = True
      # end if
    # end while
    print( "Iteration: {: 7d}, dJ = {:.4e}".format( n_iter, J - Jn ) )

    return [ w, b, J, n_iter ]
  # end def gradient_descent

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  '''.'''
  def W0( self, generator = "random" ):
    if generator == "random":
      return numpy.asmatrix( numpy.random.rand( 1, self.m_N ) )
    else:
      return numpy.asmatrix( numpy.zeros( ( 1, self.m_N ) ) )
    # end if
  # end def W0

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  '''.'''
  def B0( self, generator = "random" ):
    if generator == "random":
      return random.uniform( -1, 1 )
    else:
      return 0.0
    # end if
  # end def B0

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  '''.'''
  def X( self ):
    return self.m_X
  # end def

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  '''.'''
  def Y( self ):
    return self.m_Y
  # end def

# end class

## eof - $RCSfile$
