## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import random, sys
from LinearCost import *

import matplotlib.pyplot as plt
import matplotlib.tri as tri

# -- Read data
if len( sys.argv ) < 7:
  print( "Usage: [python3] {:s} data_file [zeros/random] scale [none/standardize/decorrelate] epsilon alpha".format( sys.argv[ 0 ] ) )
  sys.exit( 1 )
# end if
init_values = sys.argv[ 2 ]
init_scale = float( sys.argv[ 3 ] )
normalization = sys.argv[ 4 ]
epsilon = float( sys.argv[ 5 ] )
alpha = float( sys.argv[ 6 ] )
cost = LinearCost.ReadFromFile( sys.argv[ 1 ], normalization = normalization )

# -- Train weights and bias
w0 = cost.W0( init_values ) * init_scale
b0 = cost.B0( init_values ) * init_scale
[ wG, bG, JG, n_iter, v_params ] = cost.gradient_descent( w0, b0, alpha, epsilon )
print( "\n-- Gradient descent  --" )
print( "Starting parameters:", w0, b0 )
print( "w =", wG )
print( "b =", bG )
print( "J =", JG )
print( "Iterations =", n_iter )

[ wA, bA, JA ] = cost.analytic_solve( )
print( "\n-- Analytic solution --" )
print( "w =", wA )
print( "b =", bA )
print( "J =", JA )

# -- Prepare graphics
if cost.number_of_variables( ) == 1:
  min_w = -5.0
  min_b = -10.0
  max_w = 5.0
  max_b = 10.0
  sample_w = 100
  sample_b = 100
  nlevels = 50

  gw = [ ( ( ( max_w - min_w ) * float( i ) / float( sample_w ) ) + min_w ) for i in range( sample_w ) ]
  gb = [ ( ( ( max_b - min_b ) * float( i ) / float( sample_b ) ) + min_b ) for i in range( sample_b ) ]
  gz = []
  for iw in gw:
    z = []
    for ib in gb:
      z.append( cost( iw, ib ) )
    # end for
    gz.append( z )
  # end for

  fig, ax1 = plt.subplots( nrows = 1 )
  ax1.axis( "equal" )
  ax1.contour( gb, gw, gz, levels = nlevels, linewidths = 0.5, colors='k' )
  cntr1 = ax1.contourf( gb, gw, gz, levels=nlevels, cmap="Accent" )

  fig.colorbar(cntr1, ax=ax1)
  ax1.plot( [ b0 ], [ w0[ 0, 0 ] ], 'bv', ms=10)
  ax1.plot( [ bA ], [ wA[ 0, 0 ] ], 'kx', ms=10)
  ax1.plot( [ bG ], [ wG[ 0, 0 ] ], 'r+', ms=10)
  data = numpy.array( v_params )
  sx, sy = numpy.array( v_params ).T
  plt.scatter( sy, sx, c = "#ff0000", marker = "x" )
  plt.plot( sy, sx, "b--" )
  plt.show( )
# end if

## eof - $RCSfile$
