## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import numpy
import PUJ.Model.Linear

m = PUJ.Model.Linear( [ 1, 2 ] )
n = PUJ.Model.Linear( [ 2 ], 1 )

x = numpy.random.rand( 30, 1 )

print( m( 3 ) )
print( n( 3 ) )
print( m( x ) )

## eof - $RCSfile$
