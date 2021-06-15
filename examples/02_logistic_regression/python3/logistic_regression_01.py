## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, random, sys

## -------------------------------------------------------------------------
def ReadPGM( filename ):
  stream = open( filename )
  assert stream.readline( ) == 'P2\n'

  ( width, height ) = [ int( i ) for i in stream.readline( ).split( ) ]
  depth = int( stream.readline( ) )
  assert depth <= 255

  pgm_buffer = []
  for y in range( height ):
    row = []
    for x in range( width ):
      row.append( int( stream.readline( ) ) )
    # end for
    pgm_buffer.append( row )
  # end for
  stream.close( )
  return pgm_buffer
# end def

## -------------------------------------------------------------------------
def SavePGM( filename, pgm_buffer ):
  ( width, height ) = [ len( pgm_buffer ), len( pgm_buffer[ 0 ] ) ]
  stream = open( filename, 'w' )
  stream.write( 'P2\n' + str( width ) + ' ' + str( height ) + '\n255\n' )

  for x in range( width ):
    for y in range( height ):
      stream.write( str( pgm_buffer[ x ][ y ] ) + '\n' )
    # end for
  # end for
  stream.close( )
# end def

## -------------------------------------------------------------------------
def Sigmoid( z ):
  if -10 <= z and z <= 10:
    return 1.0 / ( 1.0 + math.exp( -float( z ) ) )
  elif z < -10:
    return 0.0
  else:
    return 1.0
  # end if
# end def

## -------------------------------------------------------------------------
def CostSigmoid( W, b, X, Y ):
  m = len( X )
  n = len( X[ 0 ] )
  J = 0.0
  for i in range( m ):
    z = float( b )
    for j in range( n ):
      z += float( W[ j ] ) * float( X[ i ][ j ] )
    # end for
    h = Sigmoid( z )
    if Y[ i ] == 1:
      J -= math.log( h + 1e-8 )
    else:
      J -= math.log( 1.0 - h + 1e-8 )
    # end if
  # end for
  return J / float( m )
# end def

## -------------------------------------------------------------------------
def WDerivativeSigmoid( W, b, X, Y ):
  m = len( X )
  n = len( X[ 0 ] )
  dw = [ 0.0 for i in range( len( X[ 0 ] ) ) ]

  for i in range( m ):
    z = float( b )
    for j in range( n ):
      z += float( W[ j ] ) * float( X[ i ][ j ] )
    # end for
    h = Sigmoid( z )

    for j in range( n ):
      dw[ j ] -= float( X[ i ][ j ] ) * ( float( Y[ i ] ) - h )
    # end for
  # end for

  for j in range( n ):
    dw[ j ] /= float( m )
  # end for
  return dw
# end def

## -------------------------------------------------------------------------
def BDerivativeSigmoid( W, b, X, Y ):
  m = len( X )
  n = len( X[ 0 ] )
  db = 0.0
  for i in range( m ):
    z = float( b )
    for j in range( n ):
      z += float( W[ j ] ) * float( X[ i ][ j ] )
    # end for
    h = Sigmoid( z )
    db -= float( Y[ i ] ) - h
  # end for
  return db / float( m )
# end def

## -------------------------------------------------------------------------
def Tanh( z ):
  return math.tanh( float( z ) )
# end def

## -------------------------------------------------------------------------
def CostTanh( W, b, X, Y ):
  J = 0.0
  return J
# end def

## -------------------------------------------------------------------------
def WDerivativeTanh( W, b, X, Y ):
  dw = [ 0.0 for i in range( len( X[ 0 ] ) ) ]
  return dw
# end def

## -------------------------------------------------------------------------
def BDerivativeTanh( W, b, X, Y ):
  db = 0.0
  return db
# end def

## -------------------------------------------------------------------------
def TrainLogisticRegression( X, Y, a = 1e-1, e = 1e-8 ):
  n = len( X[ 0 ] )
  W = [ random.uniform( -1, 1 ) for i in range( len( X[ 0 ] ) ) ]
  b = random.uniform( -1, 1 )
  J = CostSigmoid( W, b, X, Y )
  dJ = math.inf
  i = 0

  while e < dJ:
    dw = WDerivativeSigmoid( W, b, X, Y )
    db = BDerivativeSigmoid( W, b, X, Y )

    for j in range( n ):
      W[ j ] -= a * dw[ j ]
    # end for
    b -= a * db

    Jn = CostSigmoid( W, b, X, Y )
    dJ = J - Jn

    print( 'Iteration:', i, ': dJ =', dJ )
    
    J = Jn
    i += 1
  # end while

  return [ W, b ]
# end def

## -------------------------------------------------------------------------
def EvalLogisticRegression( W, b, x ):
  assert len( W ) == len( x )

  z = b
  for i in range( len( W ) ):
    z += W[ i ] * x[ i ]
  # end for

  return Sigmoid( z )

# end def

## -------------------------------------------------------------------------
if len( sys.argv ) < 3:
  print( "Usage:", sys.argv[ 0 ], "input_pgm_file output_pgm_file" )
  sys.exit( 1 )
# end if

# Read an image and convert it to examples
pgm_image = ReadPGM( sys.argv[ 1 ] )
X = []
Y = []
for i in range( len( pgm_image ) ):
  for j in range( len( pgm_image[ i ] ) ):
    X.append( [ float( i ), float( j ) ] )
    Y.append( float( pgm_image[ i ][ j ] ) )
  # end for
# end for

# Train parameters
[ W, b ] = TrainLogisticRegression( X, Y, a = 1e-4 )

# Use parameters
for i in range( len( pgm_image ) ):
  for j in range( len( pgm_image[ i ] ) ):
    x = [ float( i ), float( j ) ]
    h = EvalLogisticRegression( W, b, x )
    pgm_image[ i ][ j ] = int( 255.0 * h )
  # end for
# end for

# Save results
SavePGM( sys.argv[ 2 ], pgm_image )

## eof - logistic_regression_01.py
