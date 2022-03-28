## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .GradientDescent import *

class ADAM( GradientDescent ):

  '''
  '''
  m_Beta1 = 0.9
  m_Beta2 = 0.999

  '''
  '''
  def __init__( self, cost ):
    super( ).__init__( cost )
  # end def

  def firstDamping( self ):
    return self.m_Beta1
  # end def

  def secondDamping( self ):
    return self.m_Beta1
  # end def

  def setFirstDamping( self, b ):
    self.m_Beta1 = b
  # end def

  def setSecondDamping( self, b ):
    self.m_Beta2 = b
  # end def

  def fit( self ):
    # Initial values
    N = self.m_Cost.model( ).numberOfParameters( )
    m = numpy.zeros( ( N, 1 ) )
    v = numpy.zeros( ( N, 1 ) )
    a = self.m_Alpha * float( -1 )
    b1 = self.m_Beta1
    b1t = self.m_Beta1
    cb1 = float( 1 ) - self.m_Beta1
    b2 = self.m_Beta2
    b2t = self.m_Beta2
    cb2 = float( 1 ) - self.m_Beta2
    [ J0, g ] = self.m_Cost.evaluate( batch_id = -1, need_gradient = False )

    # Prepare loop
    stop = False
    self.m_Iteration = 0

    # Main loop
    while not stop:

      # Perform one batch loop
      for b in range( self.m_Cost.numberOfBatches( ) ):
        [ J, g ] = self.evaluate( b )

        # Update moments estimate
        m = ( b1 * m ) + ( cb1 * g )
        v = ( b2 * v ) + ( cb2 * numpy.power( g, 2 ) )

        # Compute bias-corrected moments estimate and update parameters
        self.m_Cost.move(
          a *
          ( m / ( float( 1 ) - b1t ) ) /
          ( numpy.power( v / ( float( 1 ) - b2t ), 0.5 ) + self.m_Epsilon )
          )

        # Update dampings
        b1t *= b1
        b2t *= b2
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
