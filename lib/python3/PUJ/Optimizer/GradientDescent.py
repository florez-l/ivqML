## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy

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
  J = math.inf
  dJ = math.inf
  i = 0
  stop = False

  # Main loop
  while dJ > e and i < I and not stop:

    # Batch loop
    for b in range( cost.GetNumberOfBatches( ) ):

      # Compute gradients
      J, G = cost.CostAndGradient( T, b )
      if l > 0:
        if lt == 'ridge':
          J += l * T @ T.T
          G += 2.0 * l * T
        elif lt == 'lasso':
          J += l * numpy.abs( T ).sum( )
          G += l * ( T > 0 ).astype( G.dtype ).sum( )
          G -= l * ( T < 0 ).astype( G.dtype ).sum( )
        # end if
      # end if

      # Step forward
      T -= G * a

    # end for

    if not df is None:
      stop = df( cost.GetModel( ), J, dJ, i, i % ds == 0 )
    # end if

    # TODO: Stop criterion and debug
    i += 1

  # end while

# end def

## eof - $RCSfile$
