## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import csv, numpy, random, sys

class LogisticRegressionCost( object ):
  def __init__( self, X, Y, s, d ):
    self.m_M = X.shape[ 0 ]
    self.m_N = X.shape[ 1 ]
    self.m_X = X
    self.m_Y = Y
    self.m_S = s
    self.m_D = d
  # end def __init__

  def __call__( self, w, b ):
    L  = ( numpy.matrix( w ) @ self.m_X.T ) + b
    J  = numpy.log(       self.m_S( L[ ( Y != 0 ).T ] ) ).sum( )
    J += numpy.log( 1.0 - self.m_S( L[ ( Y == 0 ).T ] ) ).sum( )
    return -J / float( self.m_M )
  # end def __call__

  def gradient( self, w, b ):
    wxb = ( numpy.matrix( w ) @ self.m_X.T ) + b
    s = self.m_S( wxb ).T
    c = numpy.multiply(
        ( self.m_Y - s ) / numpy.multiply( s, ( 1.0 - s ) ),
        self.m_D( wxb ).T
        )
    return [ -numpy.multiply( self.m_X, c ).mean( axis = 0 ), -c.mean( ) ]
  # end def __call__
# end class

def Sigmoid( X ):
  return 1.0 / ( 1.0 + numpy.exp( -X ) )
# end def

def DSigmoid( X ):
  s = Sigmoid( X )
  return numpy.multiply( s, ( 1.0 - s ) )
# end def

# -- Read data and put it in numpy.matrix format
with open( sys.argv[ 1 ] ) as csv_file:
  csv_reader = csv.reader( csv_file, delimiter=',' )
  line_count = 0
  X = []
  Y = []
  for row in csv_reader:
    try:
      x = [ float( row[ i ] ) for i in range( len( row ) - 1 ) ]
      y = float( row[ len( row ) - 1 ] )
      X.append( x )
      Y.append( y )
    except Exception:
      pass
    # end try
  # end for
# end with
X = numpy.matrix( X, dtype = numpy.float128 )
Y = numpy.matrix( Y, dtype = numpy.float128 ).T

# -- Create trainer
cost = LogisticRegressionCost( X, Y, Sigmoid, DSigmoid )
J = cost( [ 0, 0 ], 0 )
[ dw, db ] = cost.gradient( [ 0, 0 ], 0 )

print( J, dw, db )

## eof - $RCSfile$
