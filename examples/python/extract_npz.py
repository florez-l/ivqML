## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, os, sys

dname = os.path.dirname( os.path.abspath( os.path.basename( sys.argv[ 1 ] ) ) )
suffix = os.path.splitext( os.path.basename( sys.argv[ 1 ] ) )[ 0 ]

data = numpy.load( sys.argv[ 1 ] )
for f in data.files:
  print( f )
  fname = os.path.join( dname, suffix + '_' + f + '.csv' )
  if len( data[ f ].shape ) == 3:
    n = 1
    for v in data[ f ].shape[ 1 : ]:
      n *= v
    # end for
    numpy.savetxt(
        fname,
        numpy.reshape( data[ f ], ( data[ f ].shape[ 0 ], n ) ),
        delimiter = ','
        )
  else:
    numpy.savetxt( fname, data[ f ], delimiter = ',' )
  # end if
# end for

## eof - $RCSfile$
