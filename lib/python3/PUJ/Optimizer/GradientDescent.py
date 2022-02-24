## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Base import *


class GradientDescent( Base ):

  '''
  '''
  def __init__( self, cost ):
    super( ).__init__( cost )
  # end def

  def Fit( self ):
    [ J0, g ] = self.m_Cost.evaluate( True )
    [ Jr, gr ] = self.regularize( )
    J0 += Jr
    g += gr
    stop = False
    self.m_RealIterations = 0
    while not stop:
      self.m_Cost.updateModel( -self.m_LearningRate * g )
      [ J1, g ] = self.m_Cost.evaluate( True )
      [ Jr, gr ] = self.regularize( )
      J1 += Jr
      g += gr
      if not self.m_DebugFunction is None:
        stop = self.m_DebugFunction( self.m_Cost.model( ), self.m_RealIterations, J1, J0 - J1, self.m_RealIterations % self.m_NumberOfDebugIterations == 0 )
      # end if
      stop = stop or self.m_RealIterations >= self.m_NumberOfIterations
      stop = stop or ( ( J0 - J1 ) < self.m_Epsilon )
      self.m_RealIterations += 1
      J0 = J1
    # end while
  # end def

## eof - $RCSfile$
