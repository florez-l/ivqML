## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

'''
@input D is a [m x n] real matrix of m examples with n measures.
'''
def SplitData( D, output_size, train_size = 1.0, test_size = 0.0 ):

  # -- Check preconditions
  assert isinstance( D, ( numpy.matrix, numpy.ndarray ) ), 'Invalid input data'
  assert isinstance( output_size, ( int ) ), 'Invalid output size'
  assert isinstance( train_size, ( float ) ), 'Invalid train size'
  assert isinstance( test_size, ( float ) ), 'Invalid test size'
  assert 0 <= train_size and train_size <= 1, 'Invalid train size (0<=s<=1)'
  assert 0 <= test_size and test_size <= 1, 'Invalid test size (0<=s<=1)'
  assert 0 < train_size + test_size, 'Both sizes cannot be 0'
  assert train_size + test_size <= 1, 'Both sizes sum should be <= 1'

  # -- Get sizes
  m = D.shape[ 0 ]
  assert output_size < D.shape[ 1 ], 'Invalid output size'
  n = D.shape[ 1 ] - output_size

  # -- Compute training, testing and validation sizes
  size_tra = int( float( m ) * train_size )
  size_tst = int( float( m ) * test_size )
  size_val = m - size_tra - size_tst

  # -- Extract X's
  X_tra = D[ : size_tra , : n ]
  X_tst = D[ size_tra : size_tra + size_tst , : n ]
  X_val = D[ size_tra + size_tst : , : n ]

  # -- Extract y's
  y_tra = D[ : size_tra , n : ]
  y_tst = D[ size_tra : size_tra + size_tst , n : ]
  y_val = D[ size_tra + size_tst : , n : ]

  return X_tra, y_tra, X_tst, y_tst, X_val, y_val
# end def

'''
'''
def CategorizeLabels( labels ):
  assert labels.shape[ 1 ] == 1, 'Invalid labels.'

  u = numpy.unique( labels.flatten( ).all( ) )
  P = numpy.eye( len( u ) )
  print( labels )
  print( P )
  print( u )
  return P[ labels.astype( numpy.ubyte ) , ].reshape( labels.shape[ 0 ], len( u ) )
# end def

'''
'''
def ConfusionMatrix( y_true, y_meas ):
  assert y_true.shape == y_meas.shape, 'Invalid shapes'

  if y_true.shape[ 1 ] == 1:
    y_true_ = CategorizeLabels( y_true )
    y_meas_ = CategorizeLabels( y_meas )
  else:
    y_true_ = y_true
    y_meas_ = y_meas
  # end if

  return y_true_.T @ y_meas_
# end def

'''
'''
def Accuracy( y_true, y_meas ):
  K = ConfusionMatrix( y_true, y_meas )
  return K, numpy.diag( K ).sum( ) / K.sum( )
# end def

## eof - $RCSfile$
