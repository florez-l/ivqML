## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

'''
'''
class Simple:

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  '''
  def __call__( self, model, i, J, dJ, show ):
    if show:
      print(
        'Iteration: {: 8d} , Cost: {:.4e} , Cost diff.: {:.4e}'.
        format( i, J, dJ )
        )
    # end if
  # end def

  '''
  '''
  def keep( self ):
    pass
  # end def

# end class

## eof - $RCSfile$
