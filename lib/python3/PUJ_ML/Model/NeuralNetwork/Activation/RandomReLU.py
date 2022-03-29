## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Base import *

'''
'''
class ReLU( Base ):

  '''
  '''
  def __init__( self, a = 1e-2 ):
    self.m_Alpha = a
  # end def

  '''
  '''
  def __call__( self, z, d = False ):
    assert False, 'Fix this'
  # end def

# end class

## eof - $RCSfile$
