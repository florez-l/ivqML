## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

## -------------------------------------------------------------------------
def Read( filename ):
  stream = open( filename )
  read_lines = stream.readlines( )
  stream.close( )

  lines = []
  for line in read_lines:
    line = line.strip( )
    if not line.startswith( '#' ):
      lines += [ line ]
    # end if
  # end for
  assert lines[ 0 ] == 'P2'

  ( width, height ) = ( int( i ) for i in lines[ 1 ].split( ) )
  depth = int( lines[ 2 ] )
  assert depth <= 255

  pgm_buffer = []
  for line in lines[ 3 : ]:
    pgm_buffer += [ int( v ) for v in line.split( ) ]
  # end for

  image = numpy.reshape( numpy.matrix( pgm_buffer ), ( width, height ) ).T
  min_v = image.min( )
  max_v = image.max( )

  return ( image - min_v ) / ( max_v - min_v )

# end def

## -------------------------------------------------------------------------
def Save( filename, pgm_buffer ):
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
def SaveAsCSV( filename, image, class0_size = -1, class1_size = -1 ):
  Z = numpy.column_stack( numpy.where( image == 0 ) )
  O = numpy.column_stack( numpy.where( image == 1 ) )

  numpy.random.shuffle( Z )
  numpy.random.shuffle( O )
  if class0_size >= 0:
    Z = Z[ 0 : class0_size, : ]
  # end if
  if class1_size >= 0:
    O = O[ 0 : class1_size, : ]
  # end if

  Z = numpy.append( Z, numpy.zeros( ( Z.shape[ 0 ], 1 ) ), axis = 1 )
  O = numpy.append( O, numpy.ones( ( O.shape[ 0 ], 1 ) ), axis = 1 )
  D = numpy.concatenate( ( Z, O ), axis = 0 )
  numpy.random.shuffle( D )

  numpy.savetxt( filename, D, delimiter = ',', fmt = '%d' )
# end def

## eof - $RCSfile$
