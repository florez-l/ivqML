## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import sys
import PGM

if len( sys.argv ) < 5:
  print( 'Usage: python ' + sys.argv[ 0 ] + ' pgm csv class0 class1' )
  sys.exit( 1 )
# end if

PGM.SaveAsCSV(
  sys.argv[ 2 ], PGM.Read( sys.argv[ 1 ] ),
  class0_size = int( sys.argv[ 3 ] ),
  class1_size = int( sys.argv[ 4 ] )
  )

## eof - $RCSfile$
