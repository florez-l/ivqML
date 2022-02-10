## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
import matplotlib.pyplot as plot
from .PlotCost import *

'''
'''
class PlotPolynomialCost( PlotCost ):

  '''
  '''
  def __init__( self, x, y ):
    super( ).__init__( nrows = 1, ncols = 2 )

    self.m_DataAxis = self.m_Axes[ 1 ]

    self.m_DataLine = None
    self.m_DataX = []
    self.m_DataAxis.scatter(
      [ x[ : , 0 ] ], [ y ], color = 'green', marker = '+'
      )

  # end def

  '''
  '''
  def __call__( self, model, i, J, dJ, show ):
    super( ).__call__( model, i, J, dJ, show )

    if show:
      # Update model
      if not isinstance( self.m_DataX, ( numpy.matrix ) ):
        n = 50
        l = self.m_DataAxis.get_xlim( )
        d = l[ 1 ] - l[ 0 ]
        self.m_DataX = \
          numpy.matrix(
            [ ( ( d * ( i / n ) ) + l[ 0 ] ) for i in range( n + 1 ) ]
          ).T
        for i in range( 1, model.numberOfParameters( ) - 1 ):
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
        self.m_DataLine, = self.m_DataAxis.plot(
          self.m_DataX[ : , 0 ], model.evaluate( self.m_DataX ),
          label = 'Iterative solution',
          color = 'blue', linewidth = 0.5
          )
      # end if
      self.m_DataLine.set_ydata( model.evaluate( self.m_DataX ) )
      self.m_DataAxis.legend( )
    # end if

  # end def

# end class

## eof - $RCSfile$
