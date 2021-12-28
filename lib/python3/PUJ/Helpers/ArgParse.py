## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse

'''
'''
class ArgParse( argparse.ArgumentParser ):

  '''
  '''
  def __init__( self ):
    super( ).__init__( )

    self.add_argument( '-a', '--alpha', type = float, default = 1e-2 )
    self.add_argument( '-b1', '--beta1', type = float, default = 0.9 )
    self.add_argument( '-b2', '--beta2', type = float, default = 0.999 )
    self.add_argument( '-l', '--regularization', type = float, default = 0 )
    self.add_argument( '-I', '--max_iterations', type = int, default = 10000 )
    self.add_argument( '-D', '--debug_step', type = int, default = 100 )
    self.add_argument( '-e', '--epsilon', type = float, default = 1e-8 )
    self.add_argument(
        '-r', '--reg_type', type = str, choices = [ 'lasso', 'ridge' ],
        default = 'ridge'
        )
    self.add_argument(
        '-t0', '--init', type = str, choices = [ 'zeros', 'ones', 'random' ],
        default = 'random'
        )
  # end def

# end class

## eof - $RCSfile$
