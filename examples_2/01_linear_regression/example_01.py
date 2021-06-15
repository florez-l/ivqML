## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, matplotlib, random, sys
import matplotlib.pyplot

# -- Get command line arguments
if len( sys.argv ) < 8:
  print( 'Usage: python ' + sys.argv[ 0 ] + ' w b r x0 x1 m alpha' )
  sys.exit( 1 )
# end if
w = float( sys.argv[ 1 ] )
b = float( sys.argv[ 2 ] )
r = float( sys.argv[ 3 ] )
x0 = float( sys.argv[ 4 ] )
x1 = float( sys.argv[ 5 ] )
m = int( sys.argv[ 6 ] )
alpha = float( sys.argv[ 7 ] )
eps = 1e-8

# -- Create data
data = []
for i in range( m + 1 ):
  x = ( ( x1 - x0 ) * float( i ) / float( m ) ) + x0
  y = ( w * x ) + b

  if r > 0:
    p = random.uniform( 0, r )
    t = random.uniform( 0, 2 * math.pi )
    x += p * math.cos( t )
    y += p * math.sin( t )
  # end if

  data += [ [ x, y ] ]
# end for
random.shuffle( data )
X = [ d[ 0 ] for d in data ]
Y = [ d[ 1 ] for d in data ]

# -- Show data
matplotlib.pyplot.ion( )
figure, axes = matplotlib.pyplot.subplots( 1, 3, sharey = False )
out_axes = axes[ 0 ]
J_axes = axes[ 1 ]
dJ_axes = axes[ 2 ]
out_axes.scatter( X, Y, marker = '+' )

# -- Prepare visual debug
x0 = min( X )
x1 = max( X )
vX = []
vY = []
for i in range( 2 * m + 1 ):
  vX += [ ( ( x1 - x0 ) * float( i ) / float( 2 * m ) ) + x0 ]
  vY += [ 0 ]
# end for

# -- Linear regression
wr = 0.0
br = 0.0
out_line, = out_axes.plot( vX, vY, color = 'red' )

J = 0.0
for i in range( len( X ) ):
  J += ( ( ( wr * X[ i ] ) + br ) - Y[ i ] ) ** 2
# end for
J /= float( len( X ) )
dJ = math.inf

JX = [ 0 ]
JY = [ J ]
J_line, = J_axes.plot( JX, JY, color = 'green' )

dJX = []
dJY = []
dJ_line = None

nIter = 0
while eps < dJ:

  if nIter % 10 == 0:
    figure.canvas.draw( )
    figure.canvas.flush_events( )
  # end if

  dw = 0.0
  db = 0.0
  for i in range( len( X ) ):
    v = ( wr * X[ i ] ) + br - Y[ i ]
    dw += X[ i ] * v
    db += v
  # end for
  dw *= 2.0 / float( len( X ) )
  db *= 2.0 / float( len( X ) )

  wr -= alpha * dw
  br -= alpha * db

  for i in range( len( vX ) ):
    vY[ i ] = ( wr * vX[ i ] ) + br
  # end for
  out_line.set_ydata( vY )

  Jn = 0.0
  for i in range( len( X ) ):
    Jn += ( ( ( wr * X[ i ] ) + br ) - Y[ i ] ) ** 2
  # end for
  Jn /= float( len( X ) )

  dJ = J - Jn
  J = Jn
  
  JX += [ len( JX ) ]
  JY += [ J ]
  J_line.set_xdata( JX )
  J_line.set_ydata( JY )
  J_axes.relim( )
  J_axes.autoscale_view( )

  dJX += [ len( JX ) ]
  dJY += [ dJ ]
  if dJ_line == None:
    dJ_line, = dJ_axes.plot( dJX, dJY, color = 'blue' )
  else:
    dJ_line.set_xdata( dJX )
    dJ_line.set_ydata( dJY )
    dJ_axes.relim( )
    dJ_axes.autoscale_view( )
  # end if

  nIter += 1

# end while

print( '*******************************************' )
print( 'Final w: {:.3e} '.format( wr ) )
print( 'Final b: {:.3e} '.format( br ) )
print( 'Final J: {:.3e} '.format( J ) )
print( 'Final dJ: {:.3e} '.format( dJ ) )
print( 'Number of iterations: ' + str( nIter ) )
print( '*******************************************' )

## eof - $RCSfile$
