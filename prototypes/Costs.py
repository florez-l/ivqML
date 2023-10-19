## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

'''
'''
class MSE:

  m_Model = None
  m_X = None
  m_Y = None

  '''
  '''
  def __init__( self, model, X, Y ):
    self.m_Model = model
    self.m_X = X
    self.m_Y = Y
  # end def

  '''
  '''
  def __call__( self ):
    d = self.m_Model( self.m_X ) - self.m_Y
    p = self.m_Model( self.m_X, True )
    return ( ( d ** 2 ).mean( ), numpy.multiply( d, p ).mean( axis = 0 ) * 2 )
  # end def

  '''
  '''
  def model( self ):
    return self.m_Model
  # end def
# end class

'''
'''
class CrossEntropy( MSE ):

  m_XY = None
  m_O = None
  m_Z = None

  '''
  '''
  def __init__( self, model, X, Y ):
    super( CrossEntropy, self ).__init__( model, X, Y )
    self.m_Z = ( self.m_Y == 0 ).nonzero( )[ 0 ]
    self.m_O = ( self.m_Y == 1 ).nonzero( )[ 0 ]
    self.m_XY = \
      self.m_X[ self.m_O , : ].sum( axis = 0 ) \
      / \
      float( self.m_X.shape[ 0 ] )
  # end def

  '''
  '''
  def __call__( self ):
    s = self.m_Model( self.m_X )
    J  = numpy.log( ( 1.0 - s[ self.m_Z , : ] ) ).sum( )
    J += numpy.log( s[ self.m_O , : ] ).sum( )

    G = numpy.zeros( self.m_Model.parameters( ).shape )
    G[ 0 , 0 ] = s.mean( ) - self.m_Y.mean( )
    G[ 0 , 1 : ] = numpy.multiply( self.m_X, s ).mean( axis = 0 ) - self.m_XY
    return ( J / float( -self.m_X.shape[ 0 ] ), G )
  # end def
# end class

## eof - $RCSfile$
