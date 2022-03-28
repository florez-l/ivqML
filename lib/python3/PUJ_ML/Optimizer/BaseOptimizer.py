## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

from abc import ABC, abstractmethod
import numpy

class BaseOptimizer( ABC ):

  '''
  '''
  m_Cost = None
  m_Epsilon = None
  m_Lambda = 0
  m_RegularizationType = 'ridge'
  m_MaximumNumberOfIterations = 10000
  m_Iteration = 0
  m_NumberOfDebugIterations = 10
  m_Debug = None

  '''
  '''
  def __init__( self, cost ):
    self.m_Cost = cost
    self.m_Epsilon = 1.0
    while self.m_Epsilon + 1 > 1:
      self.m_Epsilon /= 2
    # end while
    self.m_Epsilon *= 2
  # end def

  '''
  '''
  def setLambda( self, l ):
    self.m_Lambda = l
  # end def

  '''
  '''
  def setRegularizationToRidge( self ):
    self.m_RegularizationType = 'ridge'
  # end def

  '''
  '''
  def setRegularizationToLASSO( self ):
    self.m_RegularizationType = 'LASSO'
  # end def

  '''
  '''
  def setEpsilon( self, e ):
    self.m_Epsilon = e
  # end def

  '''
  '''
  def setMaximumNumberOfIterations( self, i ):
    self.m_MaximumNumberOfIterations = i
  # end def

  '''
  '''
  def setNumberOfDebugIterations( self, i ):
    self.m_NumberOfDebugIterations = i
  # end def

  '''
  '''
  def setDebug( self, f ):
    self.m_Debug = f
  # end def

  '''
  '''
  def iterations( self ):
    return self.m_Iteration
  # end def

  '''
  '''
  def evaluate( self, bId ):
    [ J, g ] = self.m_Cost.evaluate( batch_id = bId, need_gradient = True )
    if self.m_Lambda != 0:
      if self.m_RegularizationType == 'ridge':
        gr = self.m_Cost.model( ).parameters( ).copy( )
        J += numpy.power( g, 2 ).sum( ) * self.m_Lambda
        g += gr * ( 2 * self.m_Lambda )
      elif self.m_RegularizationType == 'LASSO':
        Jr = numpy.absolute( self.m_Cost.model( ).parameters( ) ).sum( )
        gr  = ( self.m_Cost.model( ).parameters( ) > 0 ).astype( g.dtype )
        gr -= ( self.m_Cost.model( ).parameters( ) < 0 ).astype( g.dtype )
        J += Jr * self.m_Lambda
        g += gr * self.m_Lambda
      # end if
    # end if
    return [ J, g ]
  # end def

  '''
  '''
  @abstractmethod
  def fit( self ):
    pass
  # end def
# end class

## eof - $RCSfile$
