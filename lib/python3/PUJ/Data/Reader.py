## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

'''
'''
class Reader:

  '''
  '''
  def __init__( self, train_size = 1, test_size = 0 ):

    assert 0 <= train_size and train_size <= 1, 'Invalid train size'
    assert 0 <= test_size and test_size <= 1, 'Invalid test size'

    self.m_TrainCoefficient = float( train_size )
    self.m_TestCoefficient = float( test_size )

    assert self.m_TrainCoefficient + self.m_TestCoefficient <= 1.0, \
           'Invalid sizes'

  # end def

  '''
  '''
  def FromCSV( self, filename, output_size, delimiter = ',', shuffle = False ):

    # Read and shuffle
    D = numpy.loadtxt( open( filename, 'rb' ), delimiter = delimiter )
    if shuffle:
      numpy.random.shuffle( D )
    # end if

    m = D.shape[ 0 ]
    assert output_size < D.shape[ 1 ], 'Invalid output size'
    n = D.shape[ 1 ] - output_size

    size_tra = int( float( m ) * self.m_TrainCoefficient )
    size_tst = int( float( m ) * self.m_TestCoefficient )
    size_val = m - size_tra - size_tst

    X_tra = D[ : size_tra , : n ]
    X_tst = D[ size_tra : size_tra + size_tst , : n ]
    X_val = D[ size_tra + size_tst : , : n ]

    y_tra = D[ : size_tra , n : ]
    y_tst = D[ size_tra : size_tra + size_tst , n : ]
    y_val = D[ size_tra + size_tst : , n : ]

    return ( X_tra, y_tra, X_tst, y_tst, X_val, y_val )

  # end def
  
# end class

## eof - $RCSfile$
