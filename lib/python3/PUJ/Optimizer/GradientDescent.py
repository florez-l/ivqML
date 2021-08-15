## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy

## -------------------------------------------------------------------------
def GradientDescent( cost, **kwargs ):
  a = 1e-1
  I = 1e10
  e = 1e-8
  ds = 100
  df = None
  n = cost.VectorSize( )
  t = numpy.random.rand( 1, n ) * 1e-1

  if 'learning_rate' in kwargs: a = float( kwargs[ 'learning_rate' ] )
  if 'maximum_iterations' in kwargs: I = int( kwargs[ 'maximum_iterations' ] )
  if 'epsilon' in kwargs: e = float( kwargs[ 'epsilon' ] )
  if 'debug_step' in kwargs: ds = int( kwargs[ 'debug_step' ] )
  if 'debug_function' in kwargs: df = kwargs[ 'debug_function' ]
  if 'init_theta' in kwargs:
    t0 = kwargs[ 'init_theta' ]
    if isinstance( t0, ( int, float ) ):
      t = numpy.ones( ( 1, n ) ) * float( t0 )
    elif isinstance( t0, list ):
      t = numpy.matrix( t0 )
    elif isinstance( t0, numpy.matrix ):
      t = t0
    # end if
  # end if

  # Init loop
  [ J, gt ] = cost.CostAndGradient( t )
  dJ = math.inf
  i = 0
  while dJ > e and i < I:

    # Step forward
    t -= gt * a
    [ Jn, gt ] = cost.CostAndGradient( t )
    dJ = J - Jn
    J = Jn

    # Debug
    if df != None:
      df( J, dJ, t, i, i % ds == 0 )
    # end if
    i += 1

  # end while
  if df != None:
    df( J, dJ, t, i, True )
  # end if
  
  return ( t, i )
# end def

## eof - $RCSfile$
