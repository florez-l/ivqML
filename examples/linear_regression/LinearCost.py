## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import csv, numpy, random

## -------------------------------------------------------------------------
"""A class to represent the cost of a linear regression problem"""
class LinearCost:

  '''
  Constructor method from file.
  '''
  @classmethod
  def ReadFromFile( cls, filename, normalization = "none", delimiter = "," ):
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
    return cls( X, Y, normalization = normalization )
  # end def ReadFromFile( )

  '''
  Constructor method
  @input X input examples as a numpy matrix of m x n dimensions
  @input Y input results as a numpy matrix of m x 1 dimensions
  @output An object created with some intermediary values useful for
          analytic solutions and gradient descent.
  '''
  def __init__( self, X, Y, normalization = "none" ):
    # Check inputs
    assert isinstance( X, ( list, numpy.matrix ) ), "Invalid X type."
    assert isinstance( Y, ( list, numpy.matrix ) ), "Invalid Y type."

    # Copy inputs
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

    # Compute some intermediary data
    self.m_M = self.m_X.shape[ 0 ]
    self.m_N = self.m_X.shape[ 1 ]
    self.m_A = numpy.zeros( ( self.m_N, self.m_N ) )
    for i in range( self.m_M ):
      self.m_A += self.m_X[ i, : ].T @ self.m_X[ i, : ]
    # end for
    self.m_A /= float( self.m_M )
    self.m_B = 2.0 * self.m_X.mean( axis = 0 ).T
    self.m_C = 2.0 * numpy.mean( numpy.multiply( self.m_X, self.m_Y ), axis = 0 ).T
    self.m_d = 2.0 * self.m_Y.mean( )
    self.m_e = ( self.m_Y.T @ self.m_Y ).item( ) / float( self.m_M )

    # Compute covariance matrix
    if normalization != "none":
      c = self.m_X - numpy.mean( X, axis = 0 )
      S =( c.T @ c ) / float( self.m_M - 1 )
      if normalization == "standardize":
        self.m_X = ( numpy.linalg.inv( S ) @ self.m_X.T ).T
      elif normalization == "decorrelate":
        eig_val, eig_vec = numpy.linalg.eig( S )
        print( eig_val, eig_vec )
      # end if
    # end if
  # end def __init__

  '''
  Evaluation method
  @input w parameter 1 x n vector
  @input b bias real number
  @output J(w,b)
  '''
  def __call__( self, w, b ):
    assert isinstance( b, ( int, float, numpy.float128 ) ), \
           "Invalid bias type."

    if type( w ) is int:
      _w  = numpy.matrix( [ float( w ) ] )
    elif type( w ) is float:
      _w  = numpy.matrix( [ w ] )
    elif type( w ) is list:
      _w  = numpy.matrix( w )
    elif type( w ) is numpy.matrix:
      _w  = w
    else:
      raise TypeError( 'Invalid weights type.' )
    # end if

    J  = ( _w @ ( self.m_A @ _w.T ) ).item( )
    J += b * ( _w @ self.m_B ).item( )
    J += b * b
    J -= ( _w * self.m_C ).item( )
    J -= b * self.m_d
    J += self.m_e

    return J
  # end def __call__

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
      _w  = numpy.matrix( [ float( w ) ] )
    elif type( w ) is float:
      _w  = numpy.matrix( [ w ] )
    elif type( w ) is list:
      _w  = numpy.matrix( w )
    elif type( w ) is numpy.matrix:
      _w  = w
    else:
      raise TypeError( 'Invalid weights type.' )
    # end if

    dw = ( 2.0 * ( _w @ self.m_A ) ) + ( b * self.m_B.T ) - self.m_C.T
    db = ( _w @ self.m_B ).item( ) + ( 2.0 * b ) - self.m_d

    return [ dw, db ]
  # end def gradient

  '''
  Analytic solution of the linear regression problem
  @output [ w, b ] real parameters that minimize the problem given the
          inputs X and Y.
  '''
  def analytic_solve( self ):
    A_inv = numpy.linalg.inv( self.m_A ).T
    AB = A_inv.T @ self.m_B

    num_b = ( 2.0 * self.m_d ) - ( self.m_C.T @ AB ).item( )
    den_b = 4.0 - ( self.m_B.T @ AB ).item( )
    b = num_b / den_b

    w = ( A_inv * ( self.m_C - ( b * self.m_B ) ) ).T / 2.0

    return [ w, b, self( w, b ) ]
  # end def analytic_solve

  '''.'''
  def gradient_descent( self, w0, b0, alpha, epsilon ):
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
    J = self( w, b )

    vparams = [ [ numpy.squeeze( numpy.asarray( w ) ).tolist( ), b ] ]
    n_iter = 0
    stop = False
    while not stop:
      [ dw, db ] = self.gradient( w, b )
      w = w - ( dw * alpha )
      b = b - ( db * alpha )
      Jn = self( w, b )
      vparams.append( [ numpy.squeeze( numpy.asarray( w ) ).tolist( ), b ] )

      if n_iter % 1000 == 0:
        print( "Iteration: {: 7d}, dJ = {:.4e}".format( n_iter, J - Jn ) )
      # end if
      if J - Jn < epsilon:
        stop = True
      # end if
      J = Jn
      n_iter += 1
    # end while

    return [ w, b, J, n_iter, vparams ]
  # end def gradient_descent

  '''.'''
  def W0( self, generator = "random" ):
    if generator == "random":
      return numpy.asmatrix( numpy.random.rand( 1, self.m_N ) )
    else:
      return numpy.asmatrix( numpy.zeros( ( 1, self.m_N ) ) )
    # end if
  # end def W0

  '''.'''
  def B0( self, generator = "random" ):
    if generator == "random":
      return random.uniform( -1, 1 )
    else:
      return 0.0
    # end if
  # end def B0

  '''.'''
  def number_of_variables( self ):
    return self.m_N
  # end def

  '''.'''
  def number_of_samples( self ):
    return self.m_N
  # end def

# end class

## eof - $RCSfile$
