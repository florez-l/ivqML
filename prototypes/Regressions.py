## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

'''
'''
class Linear:

  m_T = None

  '''
  '''
  def __init__( self, n = 0 ):
    self.m_T = numpy.zeros( ( 1, n + 1 ) )
  # end def

  '''
  '''
  def __call__( self, X, derivative = False ):
    if derivative:
      return numpy.concatenate( ( numpy.ones( ( X.shape[ 0 ], 1 ) ), X ), axis = 1 )
    else:
      return X @ self.m_T[ : , 1 : ].T + self.m_T[ 0 , 0 ]
    # end if
  # end def

  def __iadd__( self, d ):
    self.m_T += d
    return self
  # end def

  '''
  '''
  def __getitem__( self, k ):
    return self.m_T[ 0 , k ]
  # end def

  '''
  '''
  def __setitem__( self, k, v ):
    self.m_T[ 0 , k ] = v
  # end def

  '''
  '''
  def __str__( self ):
    s = str( self.m_T.size )
    for i in range( self.m_T.size ):
      s += ' {:.3e}'.format( self.m_T[ 0 , i ] )
    # end for
    return s
  # end def

  '''
  '''
  def parameters( self ):
    return self.m_T
  # end def

  '''
  '''
  def fit( self, X, Y, r = 2, l = 0 ):
    m = X.shape[ 0 ]
    n = X.shape[ 1 ]

    R = numpy.identity( n + 1 )
    R[ 1 : , 1 : ] = ( X.T @ X ) / float( m )
    mX = X.mean( axis = 0 )
    R[ 0, 1 : ] = mX
    R[ 1 : , 0 ] = mX.T

    if l != 0:
      if r == 2: # ridge
        L = numpy.identity( n + 1 ) * l
        L[ 0 , 0 ] = 0
        R += L
      else: # LASSO: does it have any meaning in analytical regression?
        # L = numpy.zeros( ( n + 1, n + 1 ) )
        # R += L
        pass
      # end if
    # end if

    c = numpy.zeros( ( 1, n + 1 ) )
    c[ 0, 0 ] = Y.mean( )
    c[ 0, 1 : ] = numpy.multiply( X, Y ).mean( axis = 0 )
    self.m_T = c @ numpy.linalg.inv( R )
  # end def
# end class

## eof - $RCSfile$
