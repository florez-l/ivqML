## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy

'''
'''
def Regularize( T, G, lt ):
  if lt == 'ridge':
    rJ = T @ T.T
    rG = 2.0 * T
  elif lt == 'lasso':
    rJ = numpy.abs( T ).sum( )
    rG = \
      ( T > 0 ).astype( G.dtype ).sum( ) - ( T < 0 ).astype( G.dtype ).sum( )
  # end if
  return rJ, rG
# end def

'''
'''
def GradientDescent( cost, **kwargs ):
  a = 1e-1
  l = 0.0
  lt = 'ridge'
  I = 1e10
  e = 1e-8
  ds = 100
  df = None
  T = cost.GetInitialParameters( )

  if 'learning_rate' in kwargs: a = float( kwargs[ 'learning_rate' ] )
  if 'regularization' in kwargs: l = float( kwargs[ 'regularization' ] )
  if 'reg_type' in kwargs: lt = kwargs[ 'reg_type' ]
  if 'max_iter' in kwargs: I = int( kwargs[ 'max_iter' ] )
  if 'epsilon' in kwargs: e = float( kwargs[ 'epsilon' ] )
  if 'debug_step' in kwargs: ds = int( kwargs[ 'debug_step' ] )
  if 'debug_function' in kwargs: df = kwargs[ 'debug_function' ]

  # Init loop
  J, G = cost.CostAndGradient( T )
  if l > 0:
    rJ, rG = Regularize( T, G, lt )
    J += rJ
    G += rG
  # end if
  dJ = math.inf
  i = 0
  stop = False
  while dJ > e and i < I and not stop:

    # Step forward
    T -= G * a
    Jn, G = cost.CostAndGradient( T )
    if l > 0:
      rJ, rG = Regularize( T, G, lt )
      J += rJ
      G += rG
    # end if

    # Stop criterion and debug
    dJ = J - Jn
    J = Jn
    if df != None:
      stop = df( cost.GetModel( ), J, dJ, i, i % ds == 0 )
    # end if
    i += 1

  # end while

  if df != None:
    stop = df( cost.GetModel( ), J, dJ, i, i % ds == 0 )
  # end if
# end def

## eof - $RCSfile$
