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

## eof - $RCSfile$
