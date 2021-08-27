## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

'''
'''
def ConfusionMatrix( y_true, y_meas ):
  assert y_true.shape == y_meas.shape, 'Invalid shapes'

  if y_true.shape[ 1 ] == 1:
    y_true_ = numpy.append( y_true, 1 - y_true, axis = 1 )
    y_meas_ = numpy.append( y_meas, 1 - y_meas, axis = 1 )
  else:
    y_true_ = y_true
    y_meas_ = y_meas
  # end if

  return y_true_.T @ y_meas_
# end def

'''
'''
def Accuracy( y_true, y_meas ):
  K = ConfusionMatrix( y_true, y_meas )
  return numpy.diag( K ).sum( ) / K.sum( )
# end def


## eof - $RCSfile$
