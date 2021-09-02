## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
import matplotlib.pyplot as plot
import PUJ.Model.Logistic
from .Cost import *

'''
'''
class Labeling( Cost ):

  '''
  '''
  def __init__( self, x, y, test_function = None ):
    super( ).__init__( test_function = test_function, nrows = 1, ncols = 2 )

    self.m_DataAxis = self.m_Axes[ 1 ]
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

    self.m_DataAxis.scatter(
      [ xN[ : , 0 ] ], [ xN[ : , 1 ] ],
      color = 'blue', marker = 'x', label = 'zeros'
      )
    self.m_DataAxis.scatter(
      [ xP[ : , 0 ] ], [ xP[ : , 1 ] ],
      color = 'orange', marker = '+', label = 'ones'
      )
    self.m_DataAxis.legend( )
    self.m_DataContour = None

  # end def

  '''
  '''
  def __call__( self, J, dJ, t, i, show ):
    super( ).__call__( J, dJ, t, i, show )

    if show:
      z = PUJ.Model.Logistic( t )( self.m_Data ).reshape( self.m_DX.shape )
      if self.m_DataContour != None:
        for c in self.m_DataContour.collections:
          c.remove( )
        # end for
      # end if
      self.m_DataContour = \
        self.m_DataAxis.contourf( self.m_DX, self.m_DY, z, alpha = 0.5 )
    # end if
  # end def

# end class




#   # end def

#   '''
#   '''
#   def __call__( self, J, dJ, t, i, show ):

#     self.m_CostX += [ i ]
#     self.m_CostY += [ J ]

#   # end def

#   '''
#   '''
#   def KeepFigures( self ):
#     plot.ioff( )
#     plot.show( )
#   # end def

# # end class

## eof - $RCSfile$
