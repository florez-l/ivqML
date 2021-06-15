## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

## -------------------------------------------------------------------------
def Read( filename ):
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

## eof - $RCSfile$
