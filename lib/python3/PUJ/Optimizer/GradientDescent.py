## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

class GradientDescent:

  '''
  '''
  m_Cost = None
  m_LearningRate = 1e-2
  m_Epsilon = 1e-12
  m_NumberOfIterations = 1000
  m_NumberOfDebugIterations = 10

  '''
  '''
  def __init__( self, cost ):
    self.m_Cost = cost
  # end def

  def setLearningRate( self, a ):
    self.m_LearningRate = a
  # end def

  def setEpsilon( self, e ):
    self.m_Epsilon = e
  # end def

  def setNumberOfIterations( self, i ):
    self.m_NumberOfIterations = i
  # end def

  def setNumberOfDebugIterations( self, i ):
    self.m_NumberOfDebugIterations = i
  # end def

  def Fit( self ):
    [ J0, g ] = self.m_Cost.evaluate( True )
    stop = False
    while not stop:
      print( J0 )
      self.m_Cost.updateModel( -self.m_LearningRate * g )
      [ J1, g ] = self.m_Cost.evaluate( True )
      J0 = J1
    # end while
  # end def

## eof - $RCSfile$
