## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys
sys.path.insert( 0, '../../lib/python3' )

import PUJ.Model.Linear

data = numpy.loadtxt( open( sys.argv[ 1 ], 'rb' ), delimiter = ',' )
m = PUJ.Model.Linear( numpy.matrix( data ) )
print( m )

## eof - $RCSfile$
