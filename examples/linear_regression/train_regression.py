## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import random, sys
from LinearCost import *

import matplotlib.pyplot as plt
import matplotlib.tri as tri

# -- Read data
cost = LinearCost.ReadFromFile( sys.argv[ 1 ] )
epsilon = float( sys.argv[ 2 ] )
alpha = float( sys.argv[ 3 ] )

# -- Prepare graphics
gw = [ float( i ) / 50 for i in range( -100, 100 ) ]
gb = [ float( i ) / 50 for i in range( -100, 100 ) ]
gz = []
for w in gw:
  z = []
  for b in gb:
    z.append( cost( w, b ) )
  # end for
  gz.append( z )
# end for
print( min(min(gz)), max(max(gz)) )
fig, ax1 = plt.subplots( nrows = 1 )
ax1.contour(gw, gb, gz, levels=100, linewidths=0.5, colors='k')
cntr1 = ax1.contourf(gw, gb, gz, levels=100, cmap="RdBu_r")

fig.colorbar(cntr1, ax=ax1)
#ax1.plot(gw, gb, 'ko', ms=3)
ax1.set(xlim=(-2, 2), ylim=(-2, 2))
plt.show()

# -- Train weights and bias
w0 = cost.W0( "random" )
b0 = cost.B0( "random" )
[ w, b, J, n_iter ] = cost.gradient_descent( w0, b0, alpha, epsilon )
print( "-- Gradient descent  --" )
print( "Starting parameters:", w0, b0 )
print( "w =", w )
print( "b =", b )
print( "J =", J )
print( "Iterations =", n_iter )

[ w, b, J ] = cost.analytic_solve( )
print( "\n-- Analytic solution --" )
print( "w =", w )
print( "b =", b )
print( "J =", J )

## eof - $RCSfile$
