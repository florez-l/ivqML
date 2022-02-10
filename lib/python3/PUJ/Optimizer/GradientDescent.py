## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

class GradientDescent:

  '''
  '''
  m_Cost = None
  m_LearningRate = 1e-2
  m_Epsilon = 1e-12
  m_NumberOfIterations = 1000
  m_RealIterations = 0
  m_NumberOfDebugIterations = 10
  m_DebugFunction = None

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

  def setDebugFunction( self, f ):
    self.m_DebugFunction = f
  # end def

  def realIterations( self ):
    return self.m_RealIterations
  # end def

  def Fit( self ):
    [ J0, g ] = self.m_Cost.evaluate( True )
    stop = False
    self.m_RealIterations = 0
    while not stop:
      self.m_Cost.updateModel( -self.m_LearningRate * g )
      [ J1, g ] = self.m_Cost.evaluate( True )
      if not self.m_DebugFunction is None:
        stop = self.m_DebugFunction( self.m_Cost.model( ), self.m_RealIterations, J1, J0 - J1, self.m_RealIterations % self.m_NumberOfDebugIterations == 0 )
      # end if
      stop = stop or self.m_RealIterations >= self.m_NumberOfIterations
      stop = stop or J0 < J1
      self.m_RealIterations += 1
      J0 = J1
    # end while
  # end def

## eof - $RCSfile$
