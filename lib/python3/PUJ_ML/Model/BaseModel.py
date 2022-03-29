## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

from abc import ABC, abstractmethod
import math, numpy

'''
'''
class BaseModel( ABC ):

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  Return parameters as a numpy column vector
  '''
  @abstractmethod
  def parameters( self ):
    return None
  # end def

  '''
  '''
  @abstractmethod
  def numberOfParameters( self ):
    return 0
  # end def

  '''
  Return the required number of inputs
  '''
  @abstractmethod
  def numberOfInputs( self ):
    return 0
  # end def

  '''
  Return the required number of inputs
  '''
  @abstractmethod
  def numberOfOutputs( self ):
    return 0
  # end def

  '''
  Assign parameters
  '''
  @abstractmethod
  def setParameters( self, p ):
    pass
  # end def

  '''
  Add a displacement to parameters
  '''
  @abstractmethod
  def moveParameters( self, d ):
    pass
  # end def

  '''
  Return a string-based representation of the model
  '''
  def __str__( self ):
    return ''
  # end def

  '''
  Compute the model
  '''
  def evaluate( self, X ):
    if self.numberOfParameters( ) == 0:
      raise ValueError( 'Parameters should be defined first!' )
    # end if

    rX = None
    if not isinstance( X, numpy.matrix ):
      rX = numpy.matrix( X )
    else:
      rX = X
    # end if

    if rX.shape[ 1 ] != self.numberOfInputs( ):
      raise ValueError(
        'Input size (=' + str( rX.shape[ 1 ] ) +
        ') differs from parameters (=' + str( self.numberOfInputs( ) )
        + ')'
        )
    # end if

    return self._evaluate( rX )
  # end def

  '''
  Apply a threshold when possible
  '''
  def threshold( self, X ):
    return self.evaluate( X )
  # end def

  '''
  Real evaluate method
  '''
  @abstractmethod
  def _evaluate( self, X ):
    return None
  # end def

  '''
  Cost
  '''
  class Cost( ABC ):

    '''
    Model associated to this cost
    @type Something derived from PUJ.Model.Base
    '''
    m_Model = None
    m_X = None
    m_Y = None
    m_BatchSize = 0

    '''
    Initialize an object witha zero-sized parameters vector
    '''
    def __init__( self, model, X, Y, batch_size = 0 ):
      self.m_Model = model
      self.m_X = X
      self.m_Y = Y

      self.m_BatchSize = batch_size
      if self.m_BatchSize <= 0 or self.m_BatchSize > X.shape[ 0 ]:
        self.m_BatchSize = X.shape[ 0 ]
      # end if
    # end def

    '''
    Return current model
    '''
    def model( self ):
      return self.m_Model
    # end def

    '''
    Get number of batches
    '''
    def numberOfBatches( self ):
      n = float( self.m_X.shape[ 0 ] ) / float( self.m_BatchSize )
      return int( math.ceil( n ) )
    # end def

    '''
    Get a particular batch
    '''
    def batch( self, bId ):
      i = bId * self.m_BatchSize
      j = ( bId + 1 ) * self.m_BatchSize
      if j > self.m_X.shape[ 0 ]:
        j = self.m_X.shape[ 0 ]
      # end if
      return [ self.m_X, self.m_Y ]
    # end def

    '''
    Shuffle input data
    '''
    def shuffle( self ):
      pass
    # end def

    '''
    '''
    def accuracy( self ):
      return 0.0
    # end def

    '''
    Compute the cost with current parameters
    '''
    def evaluate( self, batch_id = -1, need_gradient = False ):
      if batch_id >= 0 and batch_id < self.numberOfBatches( ):
        return self._evaluate( self.batch( batch_id ), need_gradient )
      else:
        return self._evaluate( [ self.m_X, self.m_Y ], need_gradient )
      # end if
    # end def

    '''
    Move model parameters along d
    '''
    def move( self, d ):
      self.m_Model.moveParameters( d )
    # end def

    '''
    '''
    @abstractmethod
    def _evaluate( self, samples, need_gradient ):
      return [ None, None ]
    # end def

  # end class

  def cost( self, X, Y ):
    return self.Cost( self, X, Y )
  # end def

# end class

## eof - $RCSfile$
