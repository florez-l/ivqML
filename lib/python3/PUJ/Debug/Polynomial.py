## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
import matplotlib.pyplot as plot
import PUJ.Model.Linear

'''
'''
class Polynomial:

  '''
  '''
  def __init__( self, x, y ):

    plot.ion( )
    self.m_Fig, self.m_Axes = plot.subplots( 1, 2, sharey = False )
    self.m_DataAxes = self.m_Axes[ 0 ]
    self.m_CostAxes = self.m_Axes[ 1 ]

    self.m_DataLine = None
    self.m_CostLine = None

    self.m_DataX = []
    self.m_CostX = []
    self.m_CostY = []

    self.m_DataAxes.scatter( [ x ], [ y ], color = 'red', marker = 'x' )

  # end def

  '''
  '''
  def __call__( self, J, dJ, t, i, show ):

    self.m_CostX += [ i ]
    self.m_CostY += [ dJ ]

    if show:

      # Update model
      model = PUJ.Model.Linear( t )
      if not isinstance( self.m_DataX, ( numpy.matrix ) ):
        n = 50
        l = self.m_DataAxes.get_xlim( )
        d = l[ 1 ] - l[ 0 ]
        self.m_DataX = \
          numpy.matrix(
            [ ( ( d * ( i / n ) ) + l[ 0 ] ) for i in range( n + 1 ) ]
            ).T
        for i in range( 1, model.Dimensions( ) ):
          self.m_DataX = \
            numpy.append(
              self.m_DataX,
              numpy.array(
                self.m_DataX[ : , 0 ] ) *
                numpy.array( self.m_DataX[ : , i - 1 ]
                ),
              axis = 1
              )
        # end for
        self.m_DataLine, = self.m_DataAxes.plot(
          self.m_DataX[ : , 0 ], model( self.m_DataX ),
          label = 'Iterative solution',
          color = 'blue', linewidth = 0.5
          )
      # end if
      self.m_DataLine.set_ydata( model( self.m_DataX ) )
      self.m_DataAxes.legend( )

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
