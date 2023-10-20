## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy

'''
'''
class GradientDescent:

  m_Cost = None
  m_A = 1e-2
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
    e = 10.0 ** ( math.log10( self.m_A ) * 2.0 )
    stop = False
    m = self.m_Cost.model( )
    i = 0
    while not stop:
      J, D = self.m_Cost( )
      D *= self.m_A * float( -1 )
      d = ( ( D @ D.T ) ** 0.5 ).sum( )
      stop = d <= e
      if i % 1000 == 0 or stop:
        self.m_Debug( J, d )
      # end if
      m += D
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
    self.m_A = 1e-2
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
  def fit( self ):
    e = 10.0 ** ( math.log10( self.m_A ) * 2.0 )
    stop = False
    m = self.m_Cost.model( )
    b1 = self.m_B1
    b2 = self.m_B2
    b1t = self.m_B1
    b2t = self.m_B2
    mm = numpy.zeros( m.parameters( ).shape )
    mv = numpy.zeros( m.parameters( ).shape )
    i = 0

    while not stop:
      J, D = self.m_Cost( )

      mm = ( mm * b1 ) + ( D * ( 1.0 - b1 ) )
      mv = ( mv * b2 ) + ( ( D ** 2 ) * ( 1.0 - b2 ) )
      D = numpy.multiply( \
            mm / ( 1.0 - b1t ), \
            ( mv / ( 1.0 - b2t ) ) ** -0.5 \
            ) \
          * \
          self.m_A * float( -1 )

      d = ( ( D @ D.T ) ** 0.5 ).sum( )
      stop = d <= e
      if i % 10 == 0 or stop:
        self.m_Debug( J, d )
      # end if
      m += D
      pD = D
      i += 1
      b1t *= b1
      b2t *= b2
    # end while
  # end def

# end class

## eof - $RCSfile$
