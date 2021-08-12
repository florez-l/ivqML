## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
import matplotlib.pyplot as plot
import PUJ.Model.Linear

'''
'''
class Cost:

  '''
  '''
  def __init__( self ):

    plot.ion( )
    self.m_Fig, self.m_CostAxes = plot.subplots( 1, 1, sharey = False )

    self.m_CostLine = None

    self.m_CostX = []
    self.m_CostY = []

  # end def

  '''
  '''
  def __call__( self, J, dJ, t, i, show ):

    self.m_CostX += [ i ]
    self.m_CostY += [ J ]

    if show:

      # Update cost figure
      if self.m_CostLine == None:
        self.m_CostLine, = \
          self.m_CostAxes.plot( self.m_CostX, self.m_CostY, color = 'green' )
      # end if

      self.m_CostLine.set_label( 'Cost ({:.3e})'.format( J ) )
      self.m_CostLine.set_xdata( self.m_CostX )
      self.m_CostLine.set_ydata( self.m_CostY )
      self.m_CostAxes.relim( )
      self.m_CostAxes.autoscale_view( )
      self.m_CostAxes.legend( )
    
      self.m_Fig.canvas.draw( )
      self.m_Fig.canvas.flush_events( )

    # end if

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
