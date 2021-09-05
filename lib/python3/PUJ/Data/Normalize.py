## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

'''
'''
def Nothing( X ):
  n = X.shape[ 1 ]
  return X, numpy.zeros( ( 1, n ) ), numpy.ones( ( 1, n ) )
# end def

'''
'''
def Center( X ):
  o = X.mean( axis = 0 )
  n = X.shape[ 1 ]
  return X - o, o, numpy.ones( ( 1, n ) )
# end def

'''
'''
def MinMax( X ):
  min_v = X.min( axis = 0 )
  max_v = X.max( axis = 0 )
  off_v = max_v - min_v
  return ( X - min_v ) / off_v, min_v, off_v
# end def

'''
'''
def Standardize( X ):
  o = X.mean( axis = 0 )
  C = X - o
  m = X.shape[ 0 ]
  d = numpy.diag( ( C.T @ C ) / float( m - 1 ) ) ** 0.5
  return C / d, o, d
# end def

## eof - $RCSfile$
