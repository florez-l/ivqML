## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import sys

if len( sys.argv ) < 3:
  print( 'Usage: ' + sys.argv[ 0 ] + ' input.pbm output.csv' )
  sys.exit( 1 )
# end if

f_input = open( sys.argv[ 1 ], 'r' )
input_lines = f_input.readlines( )
f_input.close( )

# -- Check magic number (file type)
if input_lines[ 0 ] != 'P1\n':
  print( 'Input does not seem to be a PBM file' )
  sys.exit( 1 )
# end if

# -- Ignore comments
i = 1
while input_lines[ i ][ 0 ] == '#':
  i += 1
# end if

# -- Get image sizes
[ w, h ] = input_lines[ i ].split( )
w = int( w )
h = int( h )

# -- Get all bytes
all_data = ''.join( input_lines[ i + 1 : ] ).replace( '\n', '' )

# -- Get data
csv_data = ''
for i in range( len( all_data ) ):
  x = i % w
  y = i // h
  csv_data += str( x ) + ',' + str( y ) + ',' +  all_data[ i ] + '\n'
# end for

csv_file = open( sys.argv[ 2 ], 'w' )
csv_file.write( csv_data )
csv_file.close( )

## eof - $RCSfile$
