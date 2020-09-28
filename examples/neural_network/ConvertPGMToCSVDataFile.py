## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, random, sys

# -- Check command line arguments
if len( sys.argv ) < 4:
  print(
      "Usage: [python3] " + sys.argv[ 0 ] +
      " file.pgm file.csv noise [sx sy ox oy t p/n0 p/n1 ...]"
      )
  sys.exit( 1 )
# end if
noise = float( sys.argv[ 3 ] )
sx = 1.0
sy = 1.0
ox = 0.0
oy = 0.0
t = 0.0
if len( sys.argv ) > 4:
  sx = float( sys.argv[ 4 ] )
# end if
if len( sys.argv ) > 5:
  sy = float( sys.argv[ 5 ] )
# end if
if len( sys.argv ) > 6:
  ox = float( sys.argv[ 6 ] )
# end if
if len( sys.argv ) > 7:
  oy = float( sys.argv[ 7 ] )
# end if
if len( sys.argv ) > 8:
  t = float( sys.argv[ 8 ] )
# end if
P = []
for i in range( 9, len( sys.argv ) ):
  P.append( float( sys.argv[ i ] ) )
# end if

# -- Read file
pgm = open( sys.argv[ 1 ], "rb" )
header = pgm.readline( ).decode( "ascii" )
assert header == "P5\n", "Invalid PGM header (!=\"P5\")."
size = pgm.readline( ).decode( "ascii" )
if size[ 0 ] == "#":
  size = pgm.readline( ).decode( "ascii" )
# end if
[ W, H ] = [ int( i ) for i in size.split( ) ]
depth = int( pgm.readline( ) )
assert depth <= 255, "Invalid PGM data depth (>255)."
A = []
L = {}
for j in range( H ):
  row = []
  for i in range( W ):
    v = ord( pgm.read( 1 ) )
    row.append( v )
    if not v in L:
      L[ v ] = [ [], len( L ) ]
    # end if
    x = float( i ) * sx
    y = float( j ) * sy
    rx = x * math.cos( t ) - y * math.sin( t ) + ox
    ry = x * math.sin( t ) + y * math.cos( t ) + oy
    r = L[ v ][ 1 ]
    #if noise < random.uniform( 0, 1 ):
    #r += 1
    # end if
    L[ v ][ 0 ].append( "{:.4f},{:.4f},{:d}".format( rx, ry, r % 2 ) )
  # end for
  A.append( row )
  # end for
# end for
pgm.close( )

# -- Shuffle examples
S = []
for label in L.keys( ):
  p = 1.0
  if L[ label ][ 1 ] < len( P ):
    p = P[ L[ label ][ 1 ] ]
  # end if
  random.shuffle( L[ label ][ 0 ] )
  if p <= 1:
    new_size = int( math.ceil( float( len( L[ label ][ 0 ] ) ) * p ) )
  else:
    new_size = int( p )
  # end if
  S += L[ label ][ 0 ][ 0: new_size ]
# end for
random.shuffle( S )

# -- Save results
csv = open( sys.argv[ 2 ], "w" )
csv.write( "x1,x2,y\n" )
for s in S:
  csv.write( s + "\n" )
# end for
csv.close( )

## eof - $RCSfile$
