## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy

'''
'''
def Adam( cost, **kwargs ):
  a = 1e-2
  b1 = 0.9
  b2 = 0.999
  l = 0.0
  lt = 'ridge'
  I = 1e10
  e = 1e-8
  ds = 100
  df = None
  T = cost.GetInitialParameters( )

  if 'alpha' in kwargs: a = float( kwargs[ 'alpha' ] )
  if 'beta1' in kwargs: b1 = float( kwargs[ 'beta1' ] )
  if 'beta2' in kwargs: b2 = float( kwargs[ 'beta2' ] )
  if 'regularization' in kwargs: l = float( kwargs[ 'regularization' ] )
  if 'reg_type' in kwargs: lt = kwargs[ 'reg_type' ]
  if 'max_iter' in kwargs: I = int( kwargs[ 'max_iter' ] )
  if 'epsilon' in kwargs: e = float( kwargs[ 'epsilon' ] )
  if 'debug_step' in kwargs: ds = int( kwargs[ 'debug_step' ] )
  if 'debug_function' in kwargs: df = kwargs[ 'debug_function' ]

  J = math.inf
  Jn = math.inf
  dJ = math.inf
  b1t = b1
  b2t = b2
  m = numpy.zeros( T.shape )
  v = numpy.zeros( T.shape )
  stop = False
  i = 0

  while not stop:

    # Advance iterations
    i += 1

    # Batch loop
    for b in range( cost.GetNumberOfBatches( ) ):

      # Compute cost and its gradient
      J, G = cost.CostAndGradient( T, b )

      # Update gradient
      m = ( m * b1 ) + ( G * ( 1 - b1 ) )
      v = ( v * b2 ) + ( numpy.power( G, 2 ) * ( 1 - b2 ) )
      nG = numpy.divide( m / ( 1 - b1t ), numpy.power( v / ( 1 - b2t ), 0.5 ) + e )

      # Step forward
      T -= nG * a
    # end for

    b1t *= b1
    b2t *= b2

    if not df is None:
      stop = df( cost.GetModel( ), J, dJ, i, i % ds == 0 )
    # end if

    stop = ( i >= I )

  # end while

# end def

## eof - $RCSfile$
