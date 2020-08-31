## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

## -------------------------------------------------------------------------
class LogisticSigmoid:

  Threshold = 0.5

  def __call__( self, X ):
    return 1.0 / ( 1.0 + numpy.exp( -X ) )
  # end def

  def derivative( self, X ):
    s = self( X )
    return numpy.multiply( s, ( 1.0 - s ) )
  # end def
# end class

## -------------------------------------------------------------------------
class TanhSigmoid:

  Threshold = 0.0

  def __call__( self, X ):
    return numpy.tanh( X )
  # end def

  def derivative( self, X ):
    s = self( X )
    return 1.0 - numpy.multiply( s, s )
  # end def
# end class

## eof - $RCSfile$
