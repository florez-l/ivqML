## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy, random, sys

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
  return 1.0 / ( 1.0 + numpy.exp( -z ) )
# end def

## -------------------------------------------------------------------------
def CostAndDerivativesSigmoid( W, b, X, Y ):
  m = float( X.shape[ 0 ] )
  S = Sigmoid( ( X @ W ) + b )
  YS = Y - S

  J  = numpy.log( 1.0 - S[ Y[ :, 0 ] == 0, : ] + 1e-8 ).sum( ) / m
  J += numpy.log( S[ Y[ :, 0 ] == 1, : ] + 1e-8 ).sum( ) / m
  
  return [ -J, -( X.T @ YS ) / m, -YS.mean( ) ]
# end def

## -------------------------------------------------------------------------
def TrainLogisticRegression( X, Y, a = 1e-1, e = 1e-8 ):
  m = X.shape[ 0 ]
  n = X.shape[ 1 ]
  W = numpy.random.rand( n, 1 )
  b = numpy.random.rand( 1, 1 )

  [ J, dW, db ] = CostAndDerivativesSigmoid( W, b, X, Y )
  dJ = math.inf
  i = 0

  while e < dJ:
    W -= a * dW
    b -= a * db
    [ Jn, dW, db ] = CostAndDerivativesSigmoid( W, b, X, Y )
    dJ = J - Jn

    print( 'Iteration:', i, ': dJ =', dJ )

    J = Jn
    i += 1
  # end while

  return [ W, b ]
# end def

## -------------------------------------------------------------------------
def EvalLogisticRegression( W, b, X ):
  return Sigmoid( ( X @ W ) + b )
# end def

## -------------------------------------------------------------------------
if len( sys.argv ) < 3:
  print( "Usage:", sys.argv[ 0 ], "input_pgm_file output_pgm_file" )
  sys.exit( 1 )
# end if

# Read an image and convert it to examples
pgm_image = ReadPGM( sys.argv[ 1 ] )
lX = []
lY = []
for i in range( len( pgm_image ) ):
  for j in range( len( pgm_image[ i ] ) ):
    lX.append( [ float( i ), float( j ) ] )
    lY.append( [ float( pgm_image[ i ][ j ] ) ] )
  # end for
# end for
X = numpy.asarray( lX )
Y = numpy.asarray( lY )

# Train parameters
[ W, b ] = TrainLogisticRegression( X, Y, a = 1e-4 )

# Use parameters
Yr = EvalLogisticRegression( W, b, X )
k = 0
for i in range( len( pgm_image ) ):
  for j in range( len( pgm_image[ i ] ) ):
    pgm_image[ i ][ j ] = int( 255.0 * Yr[ k ] )
    k += 1
  # end for
# end for

# Save results
SavePGM( sys.argv[ 2 ], pgm_image )

## eof - logistic_regression_02.py
