## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
import matplotlib.pyplot as plot

'''
'''
class Accuracy:

  '''
  '''
  def __init__( self, test_function = None, nrows = 1, ncols = 1 ):

    self.m_TestFunction = test_function

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

  # end def

  '''
  '''
  def __call__( self, J, dJ, t, i, show ):

    self.m_CostX += [ i ]
    self.m_CostY += [ J ]
    if not self.m_TestFunction is None:
      self.m_TestY += [ self.m_TestFunction.Cost( t ) ]
    # end if

    if show:

      # Update cost figure
      if self.m_CostLine == None:
        self.m_CostLine, = \
          self.m_CostAxis.plot( self.m_CostX, self.m_CostY, color = 'red' )
      # end if
      if not self.m_TestFunction is None and self.m_TestLine == None:
        self.m_TestLine, = \
          self.m_CostAxis.plot( self.m_CostX, self.m_TestY, color = 'green' )
      # end if

      self.m_CostLine.set_label( 'Training cost (dJ = {:.3e})'.format( dJ ) )
      self.m_CostLine.set_xdata( self.m_CostX )
      self.m_CostLine.set_ydata( self.m_CostY )
      if self.m_TestLine != None:
        self.m_TestLine.set_label( 'Test cost' )
        self.m_TestLine.set_xdata( self.m_CostX )
        self.m_TestLine.set_ydata( self.m_TestY )
      # end if
      self.m_CostAxis.relim( )
      self.m_CostAxis.autoscale_view( )
      self.m_CostAxis.legend( )

      self.m_Fig.canvas.draw( )
      self.m_Fig.canvas.flush_events( )
    # end if
  # end def

  '''
  '''
  def KeepFigures( self ):
    plot.ioff( )
    plot.show( )
  # end def

# end class

## eof - $RCSfile$
