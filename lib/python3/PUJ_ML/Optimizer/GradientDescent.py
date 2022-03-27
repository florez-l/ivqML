## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, math
from .BaseOptimizer import *

class GradientDescent( BaseOptimizer ):

  '''
  '''
  m_Alpha = 1e-2

  '''
  '''
  def __init__( self, cost ):
    super( ).__init__( cost )
  # end def

  def learningRate( self ):
    return self.m_Alpha
  # end def

  def setLearningRate( self, a ):
    self.m_Alpha = a
  # end def

  def fit( self ):

    # Prepare loop
    stop = False
    self.m_Iteration = 0
    [ J0, g ] = self.m_Cost.evaluate( batch_id = -1, need_gradient = False )

    # Main loop
    while not stop:

      # Perform one batch loop
      for b in range( self.m_Cost.numberOfBatches( ) ):
        [ J, g ] = self.evaluate( b )
        if not math.isnan( J ) and not math.isinf( J ):
          self.m_Cost.move( -self.m_Alpha * g )
        # end if
      # end for

      # Update stop criteria
      [ J1, g ] = self.m_Cost.evaluate( batch_id = -1, need_gradient = False )
      if not self.m_Debug is None:
        stop = stop or self.m_Debug(
          self.m_Cost.model( ),
          self.m_Iteration,
          J0, J0 - J1,
          self.m_Iteration % self.m_NumberOfDebugIterations == 0
          )
      # end if
      self.m_Iteration += 1
      stop = stop or self.m_Iteration >= self.m_MaximumNumberOfIterations
      stop = stop or math.isnan( J1 ) or math.isinf( J1 )

      if not stop:
        # Prepare next step
        stop = stop or abs( J0 - J1 ) <= self.m_Epsilon
        J0 = J1
        self.m_Cost.shuffle( )
      else:
        # Debug last iteration
        if not self.m_Debug is None:
          self.m_Debug(
            self.m_Cost.model( ), self.m_Iteration, J0, J0 - J1, True
            )
        # end if
      # end if
    # end while
  # end def
# end class

## eof - $RCSfile$
