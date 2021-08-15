## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
import matplotlib.pyplot as plot
import PUJ.Model.Linear

'''
'''
class Labeling:

  '''
  '''
  def __init__( self, x, y ):

    plot.ion( )
    self.m_Fig, self.m_Axes = plot.subplots( 1, 2, sharey = False )
    self.m_DataAxes = self.m_Axes[ 0 ]
    self.m_CostAxes = self.m_Axes[ 1 ]

    # self.m_DataLine = None
    self.m_CostLine = None

    # self.m_DataX = []
    self.m_CostX = []
    self.m_CostY = []

    min0 = x[ : , 0 ].min( ) - 1
    max0 = x[ : , 0 ].max( ) + 1
    min1 = x[ : , 1 ].min( ) - 1
    max1 = x[ : , 1 ].max( ) + 1

    self.m_DX, self.m_DY = numpy.meshgrid(
      numpy.arange( min0, max0, ( max0 - min0 ) / 200.0 ),
      numpy.arange( min1, max1, ( max1 - min1 ) / 200.0 )
      )
    self.m_Data = numpy.append(
      numpy.matrix( self.m_DX.ravel( ) ).T,
      numpy.matrix( self.m_DY.ravel( ) ).T,
      axis = 1
      )

    xN = x[ numpy.where( y[ : , 0 ] == 0 )[ 0 ] , : ]
    xP = x[ numpy.where( y[ : , 0 ] == 1 )[ 0 ] , : ]

    self.m_DataAxes.scatter(
      [ xN[ : , 0 ] ], [ xN[ : , 1 ] ],
      color = 'blue', marker = 'x', label = 'zeros'
      )
    self.m_DataAxes.scatter(
      [ xP[ : , 0 ] ], [ xP[ : , 1 ] ],
      color = 'orange', marker = '+', label = 'ones'
      )
    self.m_DataAxes.legend( )
    self.m_DataContour = None

  # end def

  '''
  '''
  def __call__( self, J, dJ, t, i, show ):

    self.m_CostX += [ i ]
    self.m_CostY += [ dJ ]

    if show:
      z = PUJ.Model.Logistic( t )( self.m_Data ).reshape( self.m_DX.shape )
      if self.m_DataContour != None:
        for c in self.m_DataContour.collections:
          c.remove( )
        # end for
      # end if
      self.m_DataContour = \
        self.m_DataAxes.contourf( self.m_DX, self.m_DY, z, alpha = 0.5 )

      # Update cost figure
      if self.m_CostLine == None:
        self.m_CostLine, = \
          self.m_CostAxes.plot( self.m_CostX, self.m_CostY, color = 'green' )
      # end if

      self.m_CostLine.set_label( 'Cost diff ({:.3e})'.format( J ) )
      self.m_CostLine.set_xdata( self.m_CostX )
      self.m_CostLine.set_ydata( self.m_CostY )
      self.m_CostAxes.relim( )
      self.m_CostAxes.autoscale_view( )
      self.m_CostAxes.legend( )

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
