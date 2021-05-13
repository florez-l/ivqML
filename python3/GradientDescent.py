## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy

## -------------------------------------------------------------------------
def Solve(
    cost_function,
    learning_rate = 1e-2,
    max_iterations = 1000,
    epsilon = 1e-8,
    debug_step = 1000
    ):
  W = numpy.random.uniform( -1e-1, 1e-1, ( 1, cost_function.VectorSize( ) ) )
  b = numpy.random.uniform( -1e-1, 1e-1, ( 1, 1 ) )
  [ J, dW, db ] = cost_function.CostAndDerivatives( W, b )

  dJ = math.inf
  i = 0
  while dJ > epsilon and i < max_iterations:
    W -= learning_rate * dW
    b -= learning_rate * db

    [ Jn, dW, db ] = cost_function.CostAndDerivatives( W, b )
    dJ = J - Jn
    if i % debug_step == 0:
      print( 'Iteration =', i, ': dJ =', dJ )
    # end if
    J = Jn
    i += 1
  # end while

  return [ W, b, i ]
# end def

## eof - $RCSfile$
