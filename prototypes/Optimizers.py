## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy

'''
'''
class GradientDescent:

  m_Cost = None
  m_A = 1e-1
  m_L = 0
  m_R = 2
  m_Debug = lambda obj, cost, diff : False

  '''
  '''
  def __init__( self, cost ):
    self.m_Cost = cost
  # end def

  '''
  '''
  def set_learning_rate( self, a ):
    self.m_A = a
  # end def
  
  '''
  '''
  def set_regularization_to_LASSO( self ):
    self.m_R = 1
  # end def

  '''
  '''
  def set_regularization_to_ridge( self ):
    self.m_R = 2
  # end def

  '''
  '''
  def set_regularization_coefficient( self, l ):
    self.m_L = l
  # end def

  '''
  '''
  def set_debug( self, d ):
    self.m_Debug = d
  # end def

  '''
  '''
  def fit( self ):
    stop = False
    m = self.m_Cost.model( )
    a = self.m_A * float( -1 )
    pD = None
    i = 0
    while not stop:
      J, D = self.m_Cost( )
      d = math.inf
      if not pD is None:
        d = ( ( pD - D ) @ ( pD - D ).T ) ** 0.5
      # end if
      if i % 1000 == 0:
        self.m_Debug( J, d )
      # end if
      stop = d < 1e-8
      m += D * a
      pD = D
      i += 1
    # end while
  # end def
# end class

'''
'''
class ADAM( GradientDescent ):

  m_B1 = 0.9
  m_B2 = 0.999

  '''
  '''
  def __init__( self, cost ):
    super( ADAM, self ).__init__( cost )
  # end def

  '''
  '''
  def set_beta1( self, b ):
    self.m_B1 = b
  # end def

  '''
  '''
  def set_beta2( self, b ):
    self.m_B2 = b
  # end def

  '''
  '''
  #def fit( self ):
  ## end def

# end class

## eof - $RCSfile$
