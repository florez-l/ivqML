## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
import matplotlib.pyplot as plot

'''
'''
class Cost:

  '''
  '''
  def __init__( self, nrows = 1, ncols = 1 ):

    # self.m_TestFunction = test_function

    plot.ion( )
    self.m_Fig, self.m_Axes = plot.subplots( nrows, ncols, sharey = False )

    if isinstance( self.m_Axes, ( numpy.ndarray ) ):
      self.m_CostAxis = self.m_Axes[ 0 ]
    else:
      self.m_CostAxis = self.m_Axes
    # end if

    self.m_CostLine = None
    self.m_TestLine = None

    self.m_CostX = []
    self.m_CostY = []
    self.m_TestY = []

    self.m_NumberOfIterations = -1

  # end def

  '''
  '''
  def GetNumberOfIterations( self ):
    return self.m_NumberOfIterations
  # end def

  '''
  '''
  def __call__( self, model, J, dJ, i, show ):
    self.m_CostX += [ i ]
    self.m_CostY += [ J ]
    self.m_NumberOfIterations = i

    if show:

      # Update cost figure
      if self.m_CostLine == None:
        self.m_CostLine, = \
          self.m_CostAxis.plot( self.m_CostX, self.m_CostY, color = 'red' )
      # end if

      self.m_CostLine.set_label( 'Training cost (J = {:.3e})'.format( J ) )
      self.m_CostLine.set_xdata( self.m_CostX )
      self.m_CostLine.set_ydata( self.m_CostY )
      self.m_CostAxis.relim( )
      self.m_CostAxis.autoscale_view( )
      self.m_CostAxis.legend( )

      self.m_Fig.canvas.draw( )
      self.m_Fig.canvas.flush_events( )
    # end if
    return False
  # end def

  '''
  '''
  def KeepFigures( self ):
    plot.ioff( )
    plot.show( )
  # end def

# end class

## eof - $RCSfile$
